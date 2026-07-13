; RUN: llc -mtriple=aarch64 -mattr=+sve2p2 < %s

;; These masked.compressstore operations could be natively supported with +sve2p2
;; (or by promoting to 32/64 bit elements + a truncstore), but currently are not
;; supported.

; XFAIL: *

define void @test_compressstore_nxv8i16(ptr %p, <vscale x 8 x i16> %vec, <vscale x 8 x i1> %mask) {
  tail call void @llvm.masked.compressstore.nxv8i16(<vscale x 8 x i16> %vec, ptr align 2 %p, <vscale x 8 x i1> %mask)
  ret void
}

define void @test_compressstore_nxv16i8(ptr %p, <vscale x 16 x i8> %vec, <vscale x 16 x i1> %mask) {
  tail call void @llvm.masked.compressstore.nxv16i8(<vscale x 16 x i8> %vec, ptr align 1 %p, <vscale x 16 x i1> %mask)
  ret void
}
