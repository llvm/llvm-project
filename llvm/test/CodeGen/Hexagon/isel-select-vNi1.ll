; RUN: llc -mtriple=hexagon < %s | FileCheck %s

; A vector of i1 always occupies all 8 bits of a predicate register, so
; v2i1/v4i1/v8i1 are only 2/4/8 bits wide and cannot be promoted to a vector of
; i32 the way the wider vector types are.  SELECT with a scalar condition and
; such a result is lowered by moving both predicates into general-purpose
; registers (C2_tfrpr), selecting there, and moving the result back
; (C2_tfrrp).  Because the condition is scalar the whole 8-bit pattern is
; picked at once, so the same sequence is used for every lane count.
;
; Before this was handled, the promotion computed a zero-element vector type
; and the vector legalizer asserted.

define void @sel_v2i1(i1 %c, <2 x i32> %a0, <2 x i32> %a1, <2 x i32> %b0, <2 x i32> %b1, ptr %p) {
; CHECK-LABEL: sel_v2i1:
; CHECK:     p[[PA:[0-9]+]] = vcmpw.gt
; CHECK:     p[[PB:[0-9]+]] = vcmpw.gt
; CHECK-DAG: r[[RA:[0-9]+]] = p[[PA]]
; CHECK-DAG: r[[RB:[0-9]+]] = p[[PB]]
; CHECK:     if (!p{{[0-9]+}}) r[[RA]] = add(r[[RB]],#0)
; CHECK:     p{{[0-9]+}} = r[[RA]]
  %pa = icmp sgt <2 x i32> %a0, %a1
  %pb = icmp sgt <2 x i32> %b0, %b1
  %s = select i1 %c, <2 x i1> %pa, <2 x i1> %pb
  %z = zext <2 x i1> %s to <2 x i32>
  store <2 x i32> %z, ptr %p
  ret void
}

define void @sel_v4i1(i1 %c, <4 x i16> %a0, <4 x i16> %a1, <4 x i16> %b0, <4 x i16> %b1, ptr %p) {
; CHECK-LABEL: sel_v4i1:
; CHECK:     p[[PA:[0-9]+]] = vcmph.gt
; CHECK:     p[[PB:[0-9]+]] = vcmph.gt
; CHECK-DAG: r[[RA:[0-9]+]] = p[[PA]]
; CHECK-DAG: r[[RB:[0-9]+]] = p[[PB]]
; CHECK:     if (!p{{[0-9]+}}) r[[RA]] = add(r[[RB]],#0)
; CHECK:     p{{[0-9]+}} = r[[RA]]
  %pa = icmp sgt <4 x i16> %a0, %a1
  %pb = icmp sgt <4 x i16> %b0, %b1
  %s = select i1 %c, <4 x i1> %pa, <4 x i1> %pb
  %z = zext <4 x i1> %s to <4 x i16>
  store <4 x i16> %z, ptr %p
  ret void
}

define void @sel_v8i1(i1 %c, <8 x i8> %a0, <8 x i8> %a1, <8 x i8> %b0, <8 x i8> %b1, ptr %p) {
; CHECK-LABEL: sel_v8i1:
; CHECK:     p[[PA:[0-9]+]] = vcmpb.gt
; CHECK:     p[[PB:[0-9]+]] = vcmpb.gt
; CHECK-DAG: r[[RA:[0-9]+]] = p[[PA]]
; CHECK-DAG: r[[RB:[0-9]+]] = p[[PB]]
; CHECK:     if (!p{{[0-9]+}}) r[[RA]] = add(r[[RB]],#0)
; CHECK:     p{{[0-9]+}} = r[[RA]]
  %pa = icmp sgt <8 x i8> %a0, %a1
  %pb = icmp sgt <8 x i8> %b0, %b1
  %s = select i1 %c, <8 x i1> %pa, <8 x i1> %pb
  %z = zext <8 x i1> %s to <8 x i8>
  store <8 x i8> %z, ptr %p
  ret void
}
