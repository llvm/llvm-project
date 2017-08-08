; RUN: llc < %s -mtriple=arm64-eabi -aarch64-neon-syntax=apple -asm-verbose=false | FileCheck %s

; Test efficient codegen of vector extends up from legal type to 128 bit
; and 256 bit vector types.

;-----
; Vectors of i16.
;-----
define <8 x i16> @func1(<8 x i8> %v0) nounwind {
; CHECK-LABEL: func1:
; CHECK-NEXT: ushll.8h  v0, v0, #0
; CHECK-NEXT: ret
  %r = zext <8 x i8> %v0 to <8 x i16>
  ret <8 x i16> %r
}

define <8 x i16> @func2(<8 x i8> %v0) nounwind {
; CHECK-LABEL: func2:
; CHECK-NEXT: sshll.8h  v0, v0, #0
; CHECK-NEXT: ret
  %r = sext <8 x i8> %v0 to <8 x i16>
  ret <8 x i16> %r
}

define <16 x i16> @func3(<16 x i8> %v0) nounwind {
; CHECK-LABEL: func3:
; CHECK-NEXT: ushll2.8h  v1, v0, #0
; CHECK-NEXT: ushll.8h  v0, v0, #0
; CHECK-NEXT: ret
  %r = zext <16 x i8> %v0 to <16 x i16>
  ret <16 x i16> %r
}

define <16 x i16> @func4(<16 x i8> %v0) nounwind {
; CHECK-LABEL: func4:
; CHECK-NEXT: sshll2.8h  v1, v0, #0
; CHECK-NEXT: sshll.8h  v0, v0, #0
; CHECK-NEXT: ret
  %r = sext <16 x i8> %v0 to <16 x i16>
  ret <16 x i16> %r
}

;-----
; Vectors of i32.
;-----

define <4 x i32> @afunc1(<4 x i16> %v0) nounwind {
; CHECK-LABEL: afunc1:
; CHECK-NEXT: ushll.4s v0, v0, #0
; CHECK-NEXT: ret
  %r = zext <4 x i16> %v0 to <4 x i32>
  ret <4 x i32> %r
}

define <4 x i32> @afunc2(<4 x i16> %v0) nounwind {
; CHECK-LABEL: afunc2:
; CHECK-NEXT: sshll.4s v0, v0, #0
; CHECK-NEXT: ret
  %r = sext <4 x i16> %v0 to <4 x i32>
  ret <4 x i32> %r
}

define <8 x i32> @afunc3(<8 x i16> %v0) nounwind {
; CHECK-LABEL: afunc3:
; CHECK-NEXT: ushll2.4s v1, v0, #0
; CHECK-NEXT: ushll.4s v0, v0, #0
; CHECK-NEXT: ret
  %r = zext <8 x i16> %v0 to <8 x i32>
  ret <8 x i32> %r
}

define <8 x i32> @afunc4(<8 x i16> %v0) nounwind {
; CHECK-LABEL: afunc4:
; CHECK-NEXT: sshll2.4s v1, v0, #0
; CHECK-NEXT: sshll.4s v0, v0, #0
; CHECK-NEXT: ret
  %r = sext <8 x i16> %v0 to <8 x i32>
  ret <8 x i32> %r
}

define <8 x i32> @bfunc1(<8 x i8> %v0) nounwind {
; CHECK-LABEL: bfunc1:
; CHECK-NEXT: ushll.8h  v0, v0, #0
; CHECK-NEXT: ushll2.4s v1, v0, #0
; CHECK-NEXT: ushll.4s  v0, v0, #0
; CHECK-NEXT: ret
  %r = zext <8 x i8> %v0 to <8 x i32>
  ret <8 x i32> %r
}

define <8 x i32> @bfunc2(<8 x i8> %v0) nounwind {
; CHECK-LABEL: bfunc2:
; CHECK-NEXT: sshll.8h  v0, v0, #0
; CHECK-NEXT: sshll2.4s v1, v0, #0
; CHECK-NEXT: sshll.4s  v0, v0, #0
; CHECK-NEXT: ret
  %r = sext <8 x i8> %v0 to <8 x i32>
  ret <8 x i32> %r
}

;-----
; Vectors of i64.
;-----

define <4 x i64> @zfunc1(<4 x i32> %v0) nounwind {
; CHECK-LABEL: zfunc1:
; CHECK-NEXT: ushll2.2d v1, v0, #0
; CHECK-NEXT: ushll.2d v0, v0, #0
; CHECK-NEXT: ret
  %r = zext <4 x i32> %v0 to <4 x i64>
  ret <4 x i64> %r
}

define <4 x i64> @zfunc2(<4 x i32> %v0) nounwind {
; CHECK-LABEL: zfunc2:
; CHECK-NEXT: sshll2.2d v1, v0, #0
; CHECK-NEXT: sshll.2d v0, v0, #0
; CHECK-NEXT: ret
  %r = sext <4 x i32> %v0 to <4 x i64>
  ret <4 x i64> %r
}

define <4 x i64> @bfunc3(<4 x i16> %v0) nounwind {
; CHECK-LABEL: func3:
; CHECK-NEXT: ushll.4s  v0, v0, #0
; CHECK-NEXT: ushll2.2d v1, v0, #0
; CHECK-NEXT: ushll.2d  v0, v0, #0
; CHECK-NEXT: ret
  %r = zext <4 x i16> %v0 to <4 x i64>
  ret <4 x i64> %r
}

define <4 x i64> @cfunc4(<4 x i16> %v0) nounwind {
; CHECK-LABEL: func4:
; CHECK-NEXT: sshll.4s  v0, v0, #0
; CHECK-NEXT: sshll2.2d v1, v0, #0
; CHECK-NEXT: sshll.2d  v0, v0, #0
; CHECK-NEXT: ret
  %r = sext <4 x i16> %v0 to <4 x i64>
  ret <4 x i64> %r
}
