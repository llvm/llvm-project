; Regression test for: store <32 x half> at align 64 must NOT emit vmem(Base+HwLen).
;
; A <32 x half> store (64 bytes) at 64-byte alignment is widened by TypeWidenVector
; to <64 x half> with mask [true*32, false*32]. LowerHvxMaskedOp's unaligned path used
; to emit both:
;   if (q_lo) vmem(r+#0) = v_lo   ; correct write
;   if (q_hi) vmem(r+#1) = v_hi   ; spurious: q_hi is always all-zeros, but Hexagon v73
;                                  ; still probes TLB at (Base+128). TLBMISS if unmapped.
;
; With the fix (StoreMemSize <= StoreAlign), the high vmem is elided.
;
; RUN: llc -mtriple=hexagon -mattr=+hvxv73,+hvx-length128b -O2 < %s | FileCheck %s

; CHECK-LABEL: half_vec_store_align64:
; Store the lower half-vector (the real data) — must be present.
; CHECK: if ({{q[0-9]+}}) vmem(r{{[0-9]+}}+#0) = v{{[0-9]+}}
; There must be NO store to r+#1 (the spurious TLB-probing vmem).
; CHECK-NOT: vmem(r{{[0-9]+}}+#1)

; CHECK-LABEL: half_vec_store_align32:
; CHECK-DAG: if ({{q[0-9]+}}) vmem(r{{[0-9]+}}+#0) = v{{[0-9]+}}
; CHECK-DAG: if ({{q[0-9]+}}) vmem(r{{[0-9]+}}+#1) = v{{[0-9]+}}

target datalayout = "e-m:e-p:32:32:32-a:0-n16:32-i64:64:64-i32:32:32-i16:16:16-i1:8:8-f32:32:32-f64:64:64-v32:32:32-v64:64:64-v512:512:512-v1024:1024:1024-v2048:2048:2048"
target triple = "hexagon-unknown-elf"

; Store 32 halfs into %out (64-byte aligned, NOT 128-byte aligned).
; The store [%out .. %out+64) fits entirely within one HVX vector slot.
; The spurious vmem at %out+128 must be absent.
define void @half_vec_store_align64(ptr %out, ptr %in) {
  %floats = load <32 x float>, ptr %in, align 128
  %halfs  = fptrunc <32 x float> %floats to <32 x half>
  store <32 x half> %halfs, ptr %out, align 64
  ret void
}

; With only 32-byte alignment, a 64-byte store may cross an HVX vector boundary.
; The high vmem at %out+128 must still be emitted.
define void @half_vec_store_align32(ptr %out, ptr %in) {
  %floats = load <32 x float>, ptr %in, align 128
  %halfs  = fptrunc <32 x float> %floats to <32 x half>
  store <32 x half> %halfs, ptr %out, align 32
  ret void
}
