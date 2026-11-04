; RUN: llc -O2 -march=hexagon -mcpu=hexagonv79 -o - %s | FileCheck %s

; Crash regression: ensures Hexagon QF helpers (usesQF*Operand) does
; not crash during codegen.

; CHECK-LABEL: qf_helpers_qf32
define <32 x float> @qf_helpers_qf32(<32 x float> %a, <32 x float> %b) {
entry:
  %mul = fmul <32 x float> %a, %b
  %add = fadd <32 x float> %mul, %a
  ret <32 x float> %add
}
