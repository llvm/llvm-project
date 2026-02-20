; NOTE: This test checks materialization of select scalar f16 constants via
; AdvSIMD MOVI, avoiding constant pools when NEON is available.
;
; RUN: llc -mtriple=aarch64-unknown-linux-gnu -O3 %s -o - | FileCheck %s --check-prefix=NEON
; RUN: llc -mtriple=aarch64-unknown-linux-gnu -mattr=-neon -O3 %s -o - | FileCheck %s --check-prefix=NONEON

define half @ret_half_0001() {
; NEON-LABEL: ret_half_0001:
; NEON:       movi    v0.4h, #1
; NEON-NEXT:  ret
;
; NONEON-LABEL: ret_half_0001:
; NONEON:       adrp
; NONEON:       ldr     h0
; NONEON-NEXT:  ret
entry:
  ret half 0xH0001
}

define half @ret_half_001f() {
; NEON-LABEL: ret_half_001f:
; NEON:       movi    v0.4h, #31
; NEON-NEXT:  ret
;
; NONEON-LABEL: ret_half_001f:
; NONEON:       adrp
; NONEON:       ldr     h0
; NONEON-NEXT:  ret
entry:
  ret half 0xH001f
}

; special case: zero should be materialized specially
define half @ret_half_0000() {
; NEON-LABEL: ret_half_0000:
; NEON:       movi    d0, #0000000000000000
; NEON-NEXT:  ret
;
; NONEON-LABEL: ret_half_0000:
; NONEON:       fmov    s0, wzr
; NONEON-NEXT:  ret
entry:
  ret half 0xH0000
}
