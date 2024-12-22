; RUN: llc -march=sparc  < %s | FileCheck %s

;; Ensure that spills and reloads work for various types on
;; sparcv8.

;; For i32/i64 tests, use an asm statement which clobbers most
;; registers to ensure the spill will happen.

; CHECK-LABEL: test_i32_spill:
; CHECK:       and %i0, %i1, %o0
; CHECK:       st %o0, [%fp+{{.+}}]
; CHECK:       add %o0, %o0, %g0
; CHECK:       ld [%fp+{{.+}}, %i0
define i32 @test_i32_spill(i32 %a, i32 %b) {
entry:
  %r0 = and i32 %a, %b
  ; The clobber list has all registers except g0/o0. (Only o0 is usable.)
  %0 = call i32 asm sideeffect "add $0,$1,%g0", "=r,0,~{i0},~{i1},~{i2},~{i3},~{i4},~{i5},~{i6},~{i7},~{g1},~{g2},~{g3},~{g4},~{g5},~{g6},~{g7},~{l0},~{l1},~{l2},~{l3},~{l4},~{l5},~{l6},~{l7},~{o1},~{o2},~{o3},~{o4},~{o5},~{o6},~{o7}"(i32 %r0)
  ret i32 %r0
}

; CHECK-LABEL: test_i64_spill:
; CHECK:       and %i0, %i2, %o0
; CHECK:       and %i1, %i3, %o1
; CHECK:       std %o0, [%fp+{{.+}}]
; CHECK:       add %o0, %o0, %g0
; CHECK:       ldd [%fp+{{.+}}, %i0
define i64 @test_i64_spill(i64 %a, i64 %b) {
entry:
  %r0 = and i64 %a, %b
  ; The clobber list has all registers except g0,g1,o0,o1. (Only o0/o1 are a usable pair)
  ; So, o0/o1 must be used.
  %0 = call i64 asm sideeffect "add $0,$1,%g0", "=r,0,~{i0},~{i1},~{i2},~{i3},~{i4},~{i5},~{i6},~{i7},~{g2},~{g3},~{g4},~{g5},~{g6},~{g7},~{l0},~{l1},~{l2},~{l3},~{l4},~{l5},~{l6},~{l7},~{o2},~{o3},~{o4},~{o5},~{o7}"(i64 %r0)
  ret i64 %r0
}

;; For float/double tests, a call is a suitable clobber as *all* FPU
;; registers are caller-save on sparcv8.

; CHECK-LABEL: test_float_spill:
; CHECK:       fadds %f1, %f0, [[R:%[f][0-31]]]
; CHECK:       st [[R]], [%fp+{{.+}}]
; CHECK:       call
; CHECK:       ld [%fp+{{.+}}, %f0
declare float @foo_float(float)
define float @test_float_spill(float %a, float %b) {
entry:
  %r0 = fadd float %a, %b
  %0 = call float @foo_float(float %r0)
  ret float %r0
}

; CHECK-LABEL: test_double_spill:
; CHECK:       faddd %f2, %f0, [[R:%[f][0-31]]]
; CHECK:       std [[R]], [%fp+{{.+}}]
; CHECK:       call
; CHECK:       ldd [%fp+{{.+}}, %f0
declare double @foo_double(double)
define double @test_double_spill(double %a, double %b) {
entry:
  %r0 = fadd double %a, %b
  %0 = call double @foo_double(double %r0)
  ret double %r0
}
