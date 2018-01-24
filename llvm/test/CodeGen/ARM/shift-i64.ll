; RUN: llc -mtriple=arm-eabi %s -o - | FileCheck %s
; RUN: llc -mtriple=armv6m-eabi %s -o - | FileCheck %s --check-prefix=EXPAND

define i64 @test_shl(i64 %val, i64 %amt) {
; CHECK-LABEL: test_shl:
; EXPAND-LABEL: test_shl:
  ; First calculate the hi part when the shift amount is small enough that it
  ; contains components from both halves. It'll be returned in r1 so that's a
  ; reasonable place for it to end up.
; CHECK: rsb [[REVERSE_SHIFT:.*]], r2, #32
; CHECK: lsr [[TMP:.*]], r0, [[REVERSE_SHIFT]]
; CHECK: orr r1, [[TMP]], r1, lsl r2

  ; Check whether the shift was in fact small (< 32 bits).
; CHECK: sub [[EXTRA_SHIFT:.*]], r2, #32
; CHECK: cmp [[EXTRA_SHIFT]], #0

  ; If not, the high part of the answer is just the low part shifted by the
  ; excess.
; CHECK: lslge r1, r0, [[EXTRA_SHIFT]]

  ; The low part is either a direct shift (1st inst) or 0. We can reuse the same
  ; NZCV.
; CHECK: lsl r0, r0, r2
; CHECK: movge r0, #0

; EXPAND:      push {[[REG:r[0-9]+]], lr}
; EXPAND-NEXT: bl __aeabi_llsl
; EXPAND-NEXT: pop {[[REG]], pc}
  %res = shl i64 %val, %amt
  ret i64 %res
}

; Explanation for lshr is pretty much the reverse of shl.
define i64 @test_lshr(i64 %val, i64 %amt) {
; CHECK-LABEL: test_lshr:
; EXPAND-LABEL: test_lshr:
; CHECK: rsb [[REVERSE_SHIFT:.*]], r2, #32
; CHECK: lsr r0, r0, r2
; CHECK: orr r0, r0, r1, lsl [[REVERSE_SHIFT]]
; CHECK: sub [[EXTRA_SHIFT:.*]], r2, #32
; CHECK: cmp [[EXTRA_SHIFT]], #0
; CHECK: lsrge r0, r1, [[EXTRA_SHIFT]]
; CHECK: lsr r1, r1, r2
; CHECK: movge r1, #0

; EXPAND:      push {[[REG:r[0-9]+]], lr}
; EXPAND-NEXT: bl __aeabi_llsr
; EXPAND-NEXT: pop {[[REG]], pc}
  %res = lshr i64 %val, %amt
  ret i64 %res
}

; One minor difference for ashr: the high bits must be "hi >> 31" if the shift
; amount is large to get the right sign bit.
define i64 @test_ashr(i64 %val, i64 %amt) {
; CHECK-LABEL: test_ashr:
; EXPAND-LABEL: test_ashr:
; CHECK: sub [[EXTRA_SHIFT:.*]], r2, #32
; CHECK: asr [[HI_TMP:.*]], r1, r2
; CHECK: lsr r0, r0, r2
; CHECK: rsb [[REVERSE_SHIFT:.*]], r2, #32
; CHECK: cmp [[EXTRA_SHIFT]], #0
; CHECK: orr r0, r0, r1, lsl [[REVERSE_SHIFT]]
; CHECK: asrge [[HI_TMP]], r1, #31
; CHECK: asrge r0, r1, [[EXTRA_SHIFT]]
; CHECK: mov r1, [[HI_TMP]]

; EXPAND:      push {[[REG:r[0-9]+]], lr}
; EXPAND-NEXT: bl __aeabi_lasr
; EXPAND-NEXT: pop {[[REG]], pc}
  %res = ashr i64 %val, %amt
  ret i64 %res
}
