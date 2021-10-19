; RUN: llc -mtriple=aarch64--linux-gnu -mattr=+sve -asm-verbose=0 < %s

; Test that scalable vectors that are smaller than the legal vector size can be
; properly widened to part vectors.

;
; Vectors that need widening
;

; For now, just check that these don't crash during legalization. Widening of
; scalable-vector INSERT_SUBVECTOR and EXTRACT_SUBVECTOR is not yet available.
define <vscale x 1 x i32> @widen_1i32(<vscale x 1 x i32> %illegal) nounwind {
  ret <vscale x 1 x i32> %illegal
}

define <vscale x 1 x double> @widen_1f64(<vscale x 1 x double> %illegal) nounwind {
  ret <vscale x 1 x double> %illegal
}
