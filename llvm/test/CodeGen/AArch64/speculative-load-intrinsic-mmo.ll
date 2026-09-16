; RUN: llc -mtriple=aarch64-unknown-linux-gnu -mattr=+sve \
; RUN:   -stop-after=finalize-isel < %s \
; RUN:   | FileCheck %s --check-prefixes=CHECK,SDAG
; RUN: llc -mtriple=aarch64-unknown-linux-gnu -mattr=+sve -global-isel \
; RUN:   -aarch64-enable-gisel-sve=1 -stop-after=irtranslator < %s \
; RUN:   | FileCheck %s --check-prefixes=CHECK,GISEL

define <4 x i32> @plain(ptr %ptr) {
; CHECK-LABEL: name: plain
; SDAG: LDRQui %0, 0 :: (load (s128) from %ir.ptr)
; GISEL: G_LOAD %0(p0) :: (load (<4 x i32>) from %ir.ptr)
  %r = call <4 x i32> (ptr, i1, ...) @llvm.speculative.load.v4i32.p0(ptr align 16 %ptr, i1 false, i64 16)
  ret <4 x i32> %r
}

define <4 x i32> @nontemporal(ptr %ptr) {
; CHECK-LABEL: name: nontemporal
; SDAG: LDRQui %0, 0 :: (non-temporal load (s128) from %ir.ptr)
; GISEL: G_LOAD %0(p0) :: (non-temporal load (<4 x i32>) from %ir.ptr)
  %r = call <4 x i32> (ptr, i1, ...) @llvm.speculative.load.v4i32.p0(ptr align 16 %ptr, i1 false, i64 16), !nontemporal !0
  ret <4 x i32> %r
}

define <4 x i32> @invariant(ptr %ptr) {
; CHECK-LABEL: name: invariant
; SDAG: LDRQui %0, 0 :: (invariant load (s128) from %ir.ptr)
; GISEL: G_LOAD %0(p0) :: (invariant load (<4 x i32>) from %ir.ptr)
  %r = call <4 x i32> (ptr, i1, ...) @llvm.speculative.load.v4i32.p0(ptr align 16 %ptr, i1 false, i64 16), !invariant.load !1
  ret <4 x i32> %r
}

; The alignment comes from the align attribute on the pointer argument, and
; defaults to 1.
define <4 x i32> @unaligned(ptr %ptr) {
; CHECK-LABEL: name: unaligned
; SDAG: LDRQui %0, 0 :: (load (s128) from %ir.ptr, align 1)
; GISEL: G_LOAD %0(p0) :: (load (<4 x i32>) from %ir.ptr, align 1)
  %r = call <4 x i32> (ptr, i1, ...) @llvm.speculative.load.v4i32.p0(ptr %ptr, i1 false, i64 16)
  ret <4 x i32> %r
}

; A scalable return type gives the memory operand a scalable size; it still has
; to be precise, and still must not be dereferenceable.
define <vscale x 4 x i32> @scalable(ptr %ptr) {
; CHECK-LABEL: name: scalable
; SDAG: LDR_ZXI %0, 0 :: (load (<vscale x 1 x s128>) from %ir.ptr)
; GISEL: G_LOAD %0(p0) :: (load (<vscale x 4 x i32>) from %ir.ptr)
  %r = call <vscale x 4 x i32> (ptr, i1, ...) @llvm.speculative.load.nxv4i32.p0(ptr align 16 %ptr, i1 false, i64 16)
  ret <vscale x 4 x i32> %r
}

!0 = !{i32 1}
!1 = !{}
