; RUN: llc -verify-machineinstrs -O3 -mcpu=pwr8 -mtriple=powerpc64le-unknown-linux-gnu < %s | FileCheck %s

; Test swap removal when a vector splat must be adjusted to make it legal.
;

; LH: 2016-11-17
;   Updated align attritue from 16 to 8 to keep swap instructions tests.
;   Changes have been made on little-endian to use lvx and stvx
;   instructions instead of lxvd2x/xxswapd and xxswapd/stxvd2x for
;   aligned vectors with elements up to 4 bytes

; Test generated from following C code:
;
; vector char vc = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};
; vector char vcr;
; vector short vs = {0, 1, 2, 3, 4, 5, 6, 7};
; vector short vsr;
; vector int vi = {0, 1, 2, 3};
; vector int vir;
;
; void cfoo ()
; {
;   vcr = (vector char){vc[5], vc[5], vc[5], vc[5], vc[5], vc[5], vc[5], vc[5],
;                       vc[5], vc[5], vc[5], vc[5], vc[5], vc[5], vc[5], vc[5]};
; }
;
; void sfoo ()
; {
;   vsr = (vector short){vs[6], vs[6], vs[6], vs[6],
;                        vs[6], vs[6], vs[6], vs[6]};
; }
;
; void ifoo ()
; {
;   vir = (vector int){vi[1], vi[1], vi[1], vi[1]};
; }

@vc = global <16 x i8> <i8 0, i8 1, i8 2, i8 3, i8 4, i8 5, i8 6, i8 7, i8 8, i8 9, i8 10, i8 11, i8 12, i8 13, i8 14, i8 15>, align 8
@vs = global <8 x i16> <i16 0, i16 1, i16 2, i16 3, i16 4, i16 5, i16 6, i16 7>, align 8
@vi = global <4 x i32> <i32 0, i32 1, i32 2, i32 3>, align 8
@vcr = common global <16 x i8> zeroinitializer, align 8
@vsr = common global <8 x i16> zeroinitializer, align 8
@vir = common global <4 x i32> zeroinitializer, align 8

; Function Attrs: nounwind
define void @cfoo() {
entry:
  %0 = load <16 x i8>, ptr @vc, align 8
  %vecinit30 = shufflevector <16 x i8> %0, <16 x i8> undef, <16 x i32> <i32 5, i32 5, i32 5, i32 5, i32 5, i32 5, i32 5, i32 5, i32 5, i32 5, i32 5, i32 5, i32 5, i32 5, i32 5, i32 5>
  store <16 x i8> %vecinit30, ptr @vcr, align 8
  ret void
}

; Function Attrs: nounwind
define void @sfoo() {
entry:
  %0 = load <8 x i16>, ptr @vs, align 8
  %vecinit14 = shufflevector <8 x i16> %0, <8 x i16> undef, <8 x i32> <i32 6, i32 6, i32 6, i32 6, i32 6, i32 6, i32 6, i32 6>
  store <8 x i16> %vecinit14, ptr @vsr, align 8
  ret void
}

; Function Attrs: nounwind
define void @ifoo() {
entry:
  %0 = load <4 x i32>, ptr @vi, align 8
  %vecinit6 = shufflevector <4 x i32> %0, <4 x i32> undef, <4 x i32> <i32 1, i32 1, i32 1, i32 1>
  store <4 x i32> %vecinit6, ptr @vir, align 8
  ret void
}

; Justification:
;  Byte splat of element 5 (BE) becomes element 15-5 = 10 (LE)
;  which becomes (10+8)%16 = 2 (LE swapped).
;
;  Halfword splat of element 6 (BE) becomes element 7-6 = 1 (LE)
;  which becomes (1+4)%8 = 5 (LE swapped).
;
;  Word splat of element 1 (BE) becomes element 3-1 = 2 (LE)
;  which becomes (2+2)%4 = 0 (LE swapped).

; CHECK-NOT: xxpermdi
; CHECK-NOT: xxswapd

; CHECK-LABEL: @cfoo
; CHECK: lxvd2x
; CHECK: vspltb {{[0-9]+}}, {{[0-9]+}}, 2
; CHECK: stxvd2x

; CHECK-LABEL: @sfoo
; CHECK: lxvd2x
; CHECK: vsplth {{[0-9]+}}, {{[0-9]+}}, 5
; CHECK: stxvd2x

; CHECK-LABEL: @ifoo
; CHECK: lxvd2x
; CHECK: xxspltw {{[0-9]+}}, {{[0-9]+}}, 0
; CHECK: stxvd2x
