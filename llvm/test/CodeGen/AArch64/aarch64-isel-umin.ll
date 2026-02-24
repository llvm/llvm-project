; RUN: llc -mtriple=aarch64-- -o - < %s | FileCheck %s --check-prefix=CHECK-SD
; RUN: llc -mtriple=aarch64-- -mattr=+cssc -o - < %s | FileCheck %s --check-prefix=CHECK-CSSC

target datalayout = "e-m:e-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128"
target triple = "aarch64-unknown-linux-gnu"

; auto icmpi64(long x0) { return x0 != 0; }
define i1 @icmpi64(i64 noundef %0) {
; CHECK-SD-LABEL: icmpi64:
; CHECK-SD:       // %bb.0: // %entry
; CHECK-SD-NEXT:    cmp x0, #0
; CHECK-SD-NEXT:    cset w0, ne
; CHECK-SD-NEXT:    ret
;
; CHECK-CSSC-LABEL: icmpi64:
; CHECK-CSSC:       // %bb.0: // %entry
; CHECK-CSSC-NEXT:    umin x0, x0, #1
; CHECK-CSSC-NEXT:    // kill: def $w0 killed $w0 killed $x0
; CHECK-CSSC-NEXT:    ret
;
entry:
  %2 = icmp ne i64 %0, 0
  ret i1 %2
}

; auto icmpi32(int x0) { return x0 != 0; }
define i1 @icmpi32(i32 noundef %0) {
; CHECK-SD-LABEL: icmpi32:
; CHECK-SD:       // %bb.0: // %entry
; CHECK-SD-NEXT:    cmp w0, #0
; CHECK-SD-NEXT:    cset w0, ne
; CHECK-SD-NEXT:    ret
;
; CHECK-CSSC-LABEL: icmpi32:
; CHECK-CSSC:       // %bb.0: // %entry
; CHECK-CSSC-NEXT:    umin w0, w0, #1
; CHECK-CSSC-NEXT:    ret
;
entry:
  %2 = icmp ne i32 %0, 0
  ret i1 %2
}

; auto icmpi16(short x0) { return x0 != 0; }
define i1 @icmpi16(i16 noundef %0) {
; CHECK-SD-LABEL: icmpi16:
; CHECK-SD:       // %bb.0: // %entry
; CHECK-SD-NEXT:    tst w0, #0xffff
; CHECK-SD-NEXT:    cset w0, ne
; CHECK-SD-NEXT:    ret
;
; CHECK-CSSC-LABEL: icmpi16:
; CHECK-CSSC:       // %bb.0: // %entry
; CHECK-CSSC-NEXT:    and	w8, w0, #0xffff
; CHECK-CSSC-NEXT:    umin w0, w8, #1
; CHECK-CSSC-NEXT:    ret
;
entry:
  %2 = icmp ne i16 %0, 0
  ret i1 %2
}

; auto icmpi8(char x0) { return x0 != 0; }
define i1 @icmpi8(i8 noundef %0) {
; CHECK-SD-LABEL: icmpi8:
; CHECK-SD:       // %bb.0: // %entry
; CHECK-SD-NEXT:    tst w0, #0xff
; CHECK-SD-NEXT:    cset w0, ne
; CHECK-SD-NEXT:    ret
;
; CHECK-CSSC-LABEL: icmpi8:
; CHECK-CSSC:       // %bb.0: // %entry
; CHECK-CSSC-NEXT:    and w8, w0, #0xff
; CHECK-CSSC-NEXT:    umin w0, w8, #1
; CHECK-CSSC-NEXT:    ret
;
entry:
  %2 = icmp ne i8 %0, 0
  ret i1 %2
}

; unsigned long icmpi64i8(char x0) { return x0 != 0; }
define i64 @icmpi64i8(i8 noundef %0) {
; CHECK-SD-LABEL: icmpi64i8:
; CHECK-SD:       // %bb.0: // %entry
; CHECK-SD-NEXT:    tst w0, #0xff
; CHECK-SD-NEXT:    cset w0, ne
; CHECK-SD-NEXT:    ret
;
; CHECK-CSSC-LABEL: icmpi64i8:
; CHECK-CSSC:       // %bb.0: // %entry
; CHECK-CSSC-NEXT:    and	w8, w0, #0xff
; CHECK-CSSC-NEXT:    umin w0, w8, #1
; CHECK-CSSC-NEXT:    ret
;
entry:
  %1 = icmp ne i8 %0, 0
  %2 = zext i1 %1 to i64
  ret i64 %2
}

; unsigned long setcc_i8_i64(char x0) { return x0 != 0; }
define i8 @setcc_i8_i64(i64 %x) {
; CHECK-SD-LABEL: setcc_i8_i64:
; CHECK-SD:       // %bb.0: // %entry
; CHECK-SD-NEXT:    cmp	x0, #0
; CHECK-SD-NEXT:    cset w0, ne
; CHECK-SD-NEXT:    ret
;
; CHECK-CSSC-LABEL: setcc_i8_i64:
; CHECK-CSSC:       // %bb.0: // %entry
; CHECK-CSSC-NEXT:    umin x0, x0, #1
; CHECK-CSSC-NEXT:    // kill: def $w0 killed $w0 killed $x0
; CHECK-CSSC-NEXT:    ret
;
entry:
  %cmp = icmp ne i64 %x, 0
  %conv = zext i1 %cmp to i8
  ret i8 %conv
}

; short setcc_i16_i32(int x0) { return x0 != 0; }
define i16 @setcc_i16_i32(i32 %x) {
; CHECK-SD-LABEL: setcc_i16_i32:
; CHECK-SD:       // %bb.0: // %entry
; CHECK-SD-NEXT:    cmp	w0, #0
; CHECK-SD-NEXT:    cset w0, ne
; CHECK-SD-NEXT:    ret
;
; CHECK-CSSC-LABEL: setcc_i16_i32:
; CHECK-CSSC:       // %bb.0: // %entry
; CHECK-CSSC-NEXT:    umin w0, w0, #1
; CHECK-CSSC-NEXT:    ret
;
entry:
  %cmp = icmp ne i32 %x, 0
  %conv = zext i1 %cmp to i16
  ret i16 %conv
}

; int setcc_i32_i64(unsigned long x0) { return x0 != 0; }
define i32 @setcc_i32_i64(i64 %x) {
; CHECK-SD-LABEL: setcc_i32_i64:
; CHECK-SD:       // %bb.0: // %entry
; CHECK-SD-NEXT:    cmp	x0, #0
; CHECK-SD-NEXT:    cset w0, ne
; CHECK-SD-NEXT:    ret
;
; CHECK-CSSC-LABEL: setcc_i32_i64:
; CHECK-CSSC:       // %bb.0: // %entry
; CHECK-CSSC-NEXT:    umin x0, x0, #1
; CHECK-CSSC-NEXT:    // kill: def $w0 killed $w0 killed $x0
; CHECK-CSSC-NEXT:    ret
;
entry:
  %cmp = icmp ne i64 %x, 0
  %conv = zext i1 %cmp to i32
  ret i32 %conv
}

; unsigned long setcc_i64_i64(unsigned long x0) { return x0 != 0; }
define i64 @setcc_i64_i64(i64 %x) {
; CHECK-SD-LABEL: setcc_i64_i64:
; CHECK-SD:       // %bb.0: // %entry
; CHECK-SD-NEXT:    cmp	x0, #0
; CHECK-SD-NEXT:    cset	w0, ne
; CHECK-SD-NEXT:    ret
;
; CHECK-CSSC-LABEL: setcc_i64_i64:
; CHECK-CSSC:       // %bb.0: // %entry
; CHECK-CSSC-NEXT:    umin	x0, x0, #1
; CHECK-CSSC-NEXT:    ret
;
entry:
  %cmp = icmp ne i64 %x, 0
  %conv = zext i1 %cmp to i64
  ret i64 %conv
}

define <2 x i1> @setcc_v2i1_v2i64(<2 x i64> %x) {
; CHECK-SD-LABEL: setcc_v2i1_v2i64:
; CHECK-SD:       // %bb.0: // %entry
; CHECK-SD-NEXT:    cmtst	v0.2d, v0.2d, v0.2d
; CHECK-SD-NEXT:    xtn	v0.2s, v0.2d
; CHECK-SD-NEXT:    ret
;
; CHECK-CSSC-LABEL: setcc_v2i1_v2i64:
; CHECK-CSSC:       // %bb.0: // %entry
; CHECK-CSSC-NEXT:    cmtst	v0.2d, v0.2d, v0.2d
; CHECK-CSSC-NEXT:    xtn	v0.2s, v0.2d
; CHECK-CSSC-NEXT:    ret
;
entry:
  %cmp = icmp ne <2 x i64> %x, zeroinitializer
  ret <2 x i1> %cmp
}

define <4 x i1> @setcc_v4i1_v4i32(<4 x i32> %x) {
; CHECK-SD-LABEL: setcc_v4i1_v4i32:
; CHECK-SD:       // %bb.0: // %entry
; CHECK-SD-NEXT:    cmtst	v0.4s, v0.4s, v0.4s
; CHECK-SD-NEXT:    xtn	v0.4h, v0.4s
; CHECK-SD-NEXT:    ret
;
; CHECK-CSSC-LABEL: setcc_v4i1_v4i32:
; CHECK-CSSC:       // %bb.0: // %entry
; CHECK-CSSC-NEXT:    cmtst	v0.4s, v0.4s, v0.4s
; CHECK-CSSC-NEXT:    xtn	v0.4h, v0.4s
; CHECK-CSSC-NEXT:    ret
;
entry:
  %cmp = icmp ne <4 x i32> %x, zeroinitializer
  ret <4 x i1> %cmp
}

define <8 x i1> @setcc_v8i1_v8i16(<8 x i16> %x) {
; CHECK-SD-LABEL: setcc_v8i1_v8i16:
; CHECK-SD:       // %bb.0: // %entry
; CHECK-SD-NEXT:    cmtst	v0.8h, v0.8h, v0.8h
; CHECK-SD-NEXT:    xtn	v0.8b, v0.8h
; CHECK-SD-NEXT:    ret
;
; CHECK-CSSC-LABEL: setcc_v8i1_v8i16:
; CHECK-CSSC:       // %bb.0: // %entry
; CHECK-CSSC-NEXT:    cmtst	v0.8h, v0.8h, v0.8h
; CHECK-CSSC-NEXT:    xtn	v0.8b, v0.8h
; CHECK-CSSC-NEXT:    ret
;
entry:
  %cmp = icmp ne <8 x i16> %x, zeroinitializer
  ret <8 x i1> %cmp
}

define <16 x i1> @setcc_v16i1_v16i8(<16 x i8> %x) {
; CHECK-SD-LABEL: setcc_v16i1_v16i8:
; CHECK-SD:       // %bb.0: // %entry
; CHECK-SD-NEXT:    cmtst	v0.16b, v0.16b, v0.16b
; CHECK-SD-NEXT:    ret
;
; CHECK-CSSC-LABEL: setcc_v16i1_v16i8:
; CHECK-CSSC:       // %bb.0: // %entry
; CHECK-CSSC-NEXT:    cmtst	v0.16b, v0.16b, v0.16b
; CHECK-CSSC-NEXT:    ret
;
entry:
  %cmp = icmp ne <16 x i8> %x, zeroinitializer
  ret <16 x i1> %cmp
}

define <2 x i8> @setcc_v2i8_v2i64(<2 x i64> %x) {
; CHECK-SD-LABEL: setcc_v2i8_v2i64:
; CHECK-SD:       // %bb.0: // %entry
; CHECK-SD-NEXT:    cmtst	v0.2d, v0.2d, v0.2d
; CHECK-SD-NEXT:    movi	v1.2s, #1
; CHECK-SD-NEXT:    xtn	v0.2s, v0.2d
; CHECK-SD-NEXT:    and	v0.8b, v0.8b, v1.8b
; CHECK-SD-NEXT:    ret
;
; CHECK-CSSC-LABEL: setcc_v2i8_v2i64:
; CHECK-CSSC:       // %bb.0: // %entry
; CHECK-CSSC-NEXT:    cmtst	v0.2d, v0.2d, v0.2d
; CHECK-CSSC-NEXT:    movi	v1.2s, #1
; CHECK-CSSC-NEXT:    xtn	v0.2s, v0.2d
; CHECK-CSSC-NEXT:    and	v0.8b, v0.8b, v1.8b
; CHECK-CSSC-NEXT:    ret
;
entry:
  %cmp = icmp ne <2 x i64> %x, zeroinitializer
  %conv = zext <2 x i1> %cmp to <2 x i8>
  ret <2 x i8> %conv
}

define <4 x i16> @setcc_v4i16_v4i32(<4 x i32> %x) {
; CHECK-SD-LABEL: setcc_v4i16_v4i32:
; CHECK-SD:       // %bb.0: // %entry
; CHECK-SD-NEXT:    cmtst	v0.4s, v0.4s, v0.4s
; CHECK-SD-NEXT:    movi	v1.4h, #1
; CHECK-SD-NEXT:    xtn	v0.4h, v0.4s
; CHECK-SD-NEXT:    and	v0.8b, v0.8b, v1.8b
; CHECK-SD-NEXT:    ret
;
; CHECK-CSSC-LABEL: setcc_v4i16_v4i32:
; CHECK-CSSC:       // %bb.0: // %entry
; CHECK-CSSC-NEXT:    cmtst	v0.4s, v0.4s, v0.4s
; CHECK-CSSC-NEXT:    movi	v1.4h, #1
; CHECK-CSSC-NEXT:    xtn	v0.4h, v0.4s
; CHECK-CSSC-NEXT:    and	v0.8b, v0.8b, v1.8b
; CHECK-CSSC-NEXT:    ret
;
entry:
  %cmp = icmp ne <4 x i32> %x, zeroinitializer
  %conv = zext <4 x i1> %cmp to <4 x i16>
  ret <4 x i16> %conv
}

define <4 x i32> @setcc_v4i32_v4i32(<4 x i32> %x) {
; CHECK-SD-LABEL: setcc_v4i32_v4i32:
; CHECK-SD:       // %bb.0: // %entry
; CHECK-SD-NEXT:    movi	v1.4s, #1
; CHECK-SD-NEXT:    cmeq	v0.4s, v0.4s, #0
; CHECK-SD-NEXT:    bic	v0.16b, v1.16b, v0.16b
; CHECK-SD-NEXT:    ret
;
; CHECK-CSSC-LABEL: setcc_v4i32_v4i32:
; CHECK-CSSC:       // %bb.0: // %entry
; CHECK-CSSC-NEXT:    movi	v1.4s, #1
; CHECK-CSSC-NEXT:    cmeq	v0.4s, v0.4s, #0
; CHECK-CSSC-NEXT:    bic	v0.16b, v1.16b, v0.16b
; CHECK-CSSC-NEXT:    ret
;
entry:
  %cmp = icmp ne <4 x i32> %x, zeroinitializer
  %conv = zext <4 x i1> %cmp to <4 x i32>
  ret <4 x i32> %conv
}

; auto icmpi128(int128 x0) { return x0 != 0; }
define i1 @icmpi128(i128 noundef %0) {
; CHECK-SD-LABEL: icmpi128:
; CHECK-SD:       // %bb.0: // %entry
; CHECK-SD-NEXT:    orr	x8, x0, x1
; CHECK-SD-NEXT:    cmp	x8, #0
; CHECK-SD-NEXT:    cset w0, ne
; CHECK-SD-NEXT:    ret
;
; CHECK-CSSC-LABEL: icmpi128:
; CHECK-CSSC:       // %bb.0: // %entry
; CHECK-CSSC-NEXT:    orr	x8, x0, x1
; CHECK-CSSC-NEXT:    umin	x0, x8, #1
; CHECK-CSSC-NEXT:    // kill: def $w0 killed $w0 killed $x0
; CHECK-CSSC-NEXT:    ret
;
entry:
  %2 = icmp ne i128 %0, 0
  ret i1 %2
}
