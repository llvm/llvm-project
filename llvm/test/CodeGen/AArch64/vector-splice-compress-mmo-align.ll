; RUN: llc -mtriple=aarch64-unknown-linux-gnu -stop-after=finalize-isel < %s | FileCheck %s

; Illegal vector ops are expanded through a stack slot allocated with the reduced
; (non-ABI) alignment: a scalable VECTOR_SPLICE in expandVectorSplice, and a
; fixed VECTOR_COMPRESS in expandVECTOR_COMPRESS. The spill/reload MMOs must use
; that slot alignment, not the type's larger natural one. Both slots here are
; 16-byte aligned, so no MMO may claim align 32 or more.
;
; SVE is enabled per-function (not via -mattr) so that the fixed-vector compress
; routes through the generic expander rather than an SVE-specific lowering.

; Exercises the two stores and the load in expandVectorSplice.
define <vscale x 32 x i16> @splice_nxv32i16(<vscale x 32 x i16> %a, <vscale x 32 x i16> %b) #0 {
; CHECK-LABEL: name: splice_nxv32i16
; CHECK:         stack:
; CHECK:           alignment: 16
; CHECK-NOT:     align 32
; CHECK-NOT:     align 64
  %res = call <vscale x 32 x i16> @llvm.vector.splice.nxv32i16(<vscale x 32 x i16> %a, <vscale x 32 x i16> %b, i32 -1)
  ret <vscale x 32 x i16> %res
}

; Exercises the passthru store and the load in expandVECTOR_COMPRESS. The
; passthru operand is needed to exercise the store.
define <8 x i32> @compress_v8i32(<8 x i32> %vec, <8 x i1> %mask, <8 x i32> %passthru) {
; CHECK-LABEL: name: compress_v8i32
; CHECK:         stack:
; CHECK:           alignment: 16
; CHECK-NOT:     align 32
; CHECK-NOT:     align 64
  %out = call <8 x i32> @llvm.experimental.vector.compress(<8 x i32> %vec, <8 x i1> %mask, <8 x i32> %passthru)
  ret <8 x i32> %out
}

attributes #0 = { "target-features"="+sve" }
