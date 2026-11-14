;; Check that reserving G2-G4 with target features also reserves the
;; G2_G3 and G4_G5 register pairs, the same as -sparc-reserve-app-registers.
; RUN: llc -mtriple=sparc -verify-machineinstrs < %s | FileCheck %s --check-prefix=DEFAULT
; RUN: llc -mtriple=sparc -mattr=+reserve-g2,+reserve-g3,+reserve-g4 -verify-machineinstrs < %s | FileCheck %s --check-prefix=RESERVED
; RUN: llc -mtriple=sparc -sparc-reserve-app-registers -verify-machineinstrs < %s | FileCheck %s --check-prefix=RESERVED

;; Clobber everything except G2_G3, G4_G5 and L6_L7 so the i64 operand has
;; to go in one of those pairs.
define void @pair() {
; DEFAULT-LABEL: pair:
; DEFAULT:         ldd [%sp], %g2
; DEFAULT-NEXT:    !NO_APP
; DEFAULT:         std %g2, [%sp]
;
; RESERVED-LABEL: pair:
; RESERVED-NOT:    %g2
; RESERVED-NOT:    %g4
; RESERVED:        ldd [%sp], %l6
; RESERVED-NOT:    %g2
; RESERVED-NOT:    %g4
; RESERVED:        std %l6, [%sp]
; RESERVED-NOT:    %g2
; RESERVED-NOT:    %g4
; RESERVED:        .size pair
entry:
  %v = tail call i64 asm sideeffect "ldd [%sp], $0", "=r,~{o0},~{o1},~{o2},~{o3},~{o4},~{o5},~{l0},~{l1},~{l2},~{l3},~{l4},~{l5},~{i0},~{i1},~{i2},~{i3},~{i4},~{i5}"()
  tail call void asm sideeffect "std $0, [%sp]", "r"(i64 %v)
  ret void
}
