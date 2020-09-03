; RUN: llc -aarch64-sve-vector-bits-min=256 < %s | FileCheck %s
; RUN: llc -aarch64-sve-vector-bits-min=512 < %s | FileCheck %s -check-prefixes=CHECK,VBITS_GE_512

target triple = "aarch64-unknown-linux-gnu"

; Currently there is no custom lowering for vector shuffles operating on types
; bigger than NEON. However, having no support opens us up to a code generator
; hang when expanding BUILD_VECTOR. Here we just validate the promblematic case
; successfully exits code generation.
define void @hang_when_merging_stores_after_legalisation(<8 x i32>* %a, <2 x i32> %b) #0 {
; CHECK-LABEL: hang_when_merging_stores_after_legalisation:
  %splat = shufflevector <2 x i32> %b, <2 x i32> undef, <8 x i32> zeroinitializer
  %interleaved.vec = shufflevector <8 x i32> %splat, <8 x i32> undef, <8 x i32> <i32 0, i32 4, i32 1, i32 5, i32 2, i32 6, i32 3, i32 7>
  store <8 x i32> %interleaved.vec, <8 x i32>* %a, align 4
  ret void
}

; NOTE: Currently all CONCAT_VECTORS get expanded so there's little point in
; validating all combinations of vector type.

define void @concat_vectors_v4i64(<2 x i64> %a, <2 x i64> %b, <4 x i64> *%c.addr) #0 {
; CHECK-LABEL: concat_vectors_v4i64:
; CHECK: stp q0, q1, [sp]
; CHECK: ptrue [[OUT_PG:p[0-9]+]].d, vl4
; CHECK: mov x[[LO_ADDR:[0-9]+]], sp
; CHECK: ld1d { z{{[0-9]+}}.d }, [[OUT_PG]]/z, [x[[LO_ADDR]]]
  %concat = shufflevector <2 x i64> %a, <2 x i64> %b, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  store <4 x i64> %concat, <4 x i64>* %c.addr
  ret void
}

define void @concat_vectors_v8i64(<4 x i64> *%a.addr, <4 x i64> *%b.addr, <8 x i64> *%c.addr) #0 {
; VBITS_GE_512-LABEL: concat_vectors_v8i64:
; VBITS_GE_512: ptrue [[IN_PG:p[0-9]+]].d, vl4
; VBITS_GE_512: ld1d { [[LO:z[0-9]+]].d }, [[IN_PG]]/z, [x0]
; VBITS_GE_512: ld1d { [[HI:z[0-9]+]].d }, [[IN_PG]]/z, [x1]
; VBITS_GE_512: mov x[[LO_ADDR:[0-9]+]], sp
; VBITS_GE_512: orr x[[HI_ADDR:[0-9]+]], x[[LO_ADDR]], #0x20
; VBITS_GE_512: st1d { [[LO]].d }, [[IN_PG]], [x[[LO_ADDR]]]
; VBITS_GE_512: st1d { [[HI]].d }, [[IN_PG]], [x[[HI_ADDR]]]
; VBITS_GE_512: ptrue [[OUT_PG:p[0-9]+]].d, vl8
; VBITS_GE_512: ld1d { z{{[0-9]+}}.d }, [[OUT_PG]]/z, [x8]
  %a = load <4 x i64>, <4 x i64>* %a.addr
  %b = load <4 x i64>, <4 x i64>* %b.addr
  %concat = shufflevector <4 x i64> %a, <4 x i64> %b, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  store <8 x i64> %concat, <8 x i64>* %c.addr
  ret void
}

attributes #0 = { nounwind "target-features"="+sve" }
