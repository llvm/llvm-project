; RUN: llc < %s -mtriple=arm64-eabi -aarch64-neon-syntax=apple -asm-verbose=false | FileCheck %s
; RUN: llc < %s -mtriple=arm64-eabi -aarch64-neon-syntax=apple -asm-verbose=false -global-isel -global-isel-abort=2 -pass-remarks-missed=gisel* 2>&1 | FileCheck %s --check-prefixes=CHECK,FALLBACK

; Test efficient codegen of vector extends up from legal type to 128 bit
; and 256 bit vector types.

;-----
; Vectors of i16.
;-----

; FALLBACK-NOT: remark:{{.*}}(in function: func1)
define <8 x i16> @func1(<8 x i8> %v0) nounwind {
; CHECK-LABEL: func1:
; CHECK-NEXT: ushll.8h  v0, v0, #0
; CHECK-NEXT: ret
  %r = zext <8 x i8> %v0 to <8 x i16>
  ret <8 x i16> %r
}

; FALLBACK-NOT: remark:{{.*}}(in function: func2)
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

; FALLBACK-NOT: remark:{{.*}}(in function: afunc1)
define <4 x i32> @afunc1(<4 x i16> %v0) nounwind {
; CHECK-LABEL: afunc1:
; CHECK-NEXT: ushll.4s v0, v0, #0
; CHECK-NEXT: ret
  %r = zext <4 x i16> %v0 to <4 x i32>
  ret <4 x i32> %r
}

; FALLBACK-NOT: remark:{{.*}}(in function: afunc2)
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

define <4 x i64> @zext_v4i8_to_v4i64(<4 x i8> %v0) nounwind {
; CHECK-LABEL: zext_v4i8_to_v4i64:
; CHECK-NEXT:    bic.4h  v0, #255, lsl #8
; CHECK-NEXT:    ushll.4s    v0, v0, #0
; CHECK-NEXT:    ushll2.2d   v1, v0, #0
; CHECK-NEXT:    ushll.2d    v0, v0, #0
; CHECK-NEXT:    ret
;
  %r = zext <4 x i8> %v0 to <4 x i64>
  ret <4 x i64> %r
}

define <4 x i64> @sext_v4i8_to_v4i64(<4 x i8> %v0) nounwind {
; CHECK-LABEL: sext_v4i8_to_v4i64:
; CHECK-NEXT:    ushll.4s    v0, v0, #0
; CHECK-NEXT:    ushll.2d    v1, v0, #0
; CHECK-NEXT:    ushll2.2d   v0, v0, #0
; CHECK-NEXT:    shl.2d  v0, v0, #56
; CHECK-NEXT:    shl.2d  v2, v1, #56
; CHECK-NEXT:    sshr.2d v1, v0, #56
; CHECK-NEXT:    sshr.2d v0, v2, #56
; CHECK-NEXT:    ret
;
  %r = sext <4 x i8> %v0 to <4 x i64>
  ret <4 x i64> %r
}

define <8 x i64> @zext_v8i8_to_v8i64(<8 x i8> %v0) nounwind {
; CHECK-LABEL: zext_v8i8_to_v8i64:
; CHECK-NEXT:    ushll.8h   v0, v0, #0
; CHECK-NEXT:    ushll2.4s  v2, v0, #0
; CHECK-NEXT:    ushll.4s   v0, v0, #0
; CHECK-NEXT:    ushll2.2d  v3, v2, #0
; CHECK-NEXT:    ushll2.2d  v1, v0, #0
; CHECK-NEXT:    ushll.2d   v2, v2, #0
; CHECK-NEXT:    ushll.2d   v0, v0, #0
; CHECK-NEXT:    ret
;
  %r = zext <8 x i8> %v0 to <8 x i64>
  ret <8 x i64> %r
}

define <8 x i64> @sext_v8i8_to_v8i64(<8 x i8> %v0) nounwind {
; CHECK-LABEL: sext_v8i8_to_v8i64:
; CHECK-NEXT:    sshll.8h   v0, v0, #0
; CHECK-NEXT:    sshll2.4s  v2, v0, #0
; CHECK-NEXT:    sshll.4s   v0, v0, #0
; CHECK-NEXT:    sshll2.2d  v3, v2, #0
; CHECK-NEXT:    sshll2.2d  v1, v0, #0
; CHECK-NEXT:    sshll.2d   v2, v2, #0
; CHECK-NEXT:    sshll.2d   v0, v0, #0
; CHECK-NEXT:    ret
;
  %r = sext <8 x i8> %v0 to <8 x i64>
  ret <8 x i64> %r
}

; Extends of vectors of i1.

define <32 x i8> @zext_v32i1(<32 x i1> %arg) {
; CHECK-LABEL: zext_v32i1:
; CHECK:         and.16b v0, v0, v2
; CHECK-NEXT:    and.16b v1, v1, v2
; CHECK-NEXT:    ret
  %res = zext <32 x i1> %arg to <32 x i8>
  ret <32 x i8> %res
}

define <32 x i8> @sext_v32i1(<32 x i1> %arg) {
; CHECK-LABEL: sext_v32i1:
; CHECK:         shl.16b v0, v0, #7
; CHECK-NEXT:    shl.16b v1, v1, #7
; CHECK-NEXT:    sshr.16b v0, v0, #7
; CHECK-NEXT:    sshr.16b v1, v1, #7
; CHECK-NEXT:    ret
;
  %res = sext <32 x i1> %arg to <32 x i8>
  ret <32 x i8> %res
}

define <64 x i8> @zext_v64i1(<64 x i1> %arg) {
; CHECK-LABEL: zext_v64i1:
; CHECK:         and.16b v0, v0, [[V4:v.+]]
; CHECK-NEXT:    and.16b v1, v1, [[V4]]
; CHECK-NEXT:    and.16b v2, v2, [[V4]]
; CHECK-NEXT:    and.16b v3, v3, [[V4]]
; CHECK-NEXT:    ret
;
  %res = zext <64 x i1> %arg to <64 x i8>
  ret <64 x i8> %res
}

define <64 x i8> @sext_v64i1(<64 x i1> %arg) {
; CHECK-LABEL: sext_v64i1:
; CHECK:         shl.16b v0, v0, #7
; CHECK-NEXT:    shl.16b v3, v3, #7
; CHECK-NEXT:    shl.16b v2, v2, #7
; CHECK-NEXT:    shl.16b [[V4:v.+]], v1, #7
; CHECK-NEXT:    sshr.16b v0, v0, #7
; CHECK-NEXT:    sshr.16b v1, v3, #7
; CHECK-NEXT:    sshr.16b v2, v2, #7
; CHECK-NEXT:    sshr.16b v3, [[V4]], #7
; CHECK-NEXT:    ret
;
  %res = sext <64 x i1> %arg to <64 x i8>
  ret <64 x i8> %res
}

define <1 x i128> @sext_v1x64(<1 x i64> %arg) {
; CHECK-LABEL: sext_v1x64:
; CHECK-NEXT:   .cfi_startproc
; CHECK-NEXT:    fmov    x8, d0
; CHECK-NEXT:    asr x1, x8, #63
  ; X0 & X1 are the real return registers, SDAG messes with v0 too for unknown reasons.
; CHECK:    {{(mov.d   v0[1], x1)?}}
; CHECK:    fmov    x0, d0
; CHECK:    ret
;
  %res = sext <1 x i64> %arg to <1 x i128>
  ret <1 x i128> %res
}
