; RUN: llc < %s -mtriple=arm64-apple-ios7.0 -aarch64-neon-syntax=apple -verify-machineinstrs

; CHECK: fpr128
; CHECK: ld1.2d
; CHECK: str q
; CHECK: inlineasm
; CHECK: ldr q
; CHECK: st1.2d
define void @fpr128(ptr %p) nounwind ssp {
entry:
  %x = load <4 x float>, ptr %p, align 16
  call void asm sideeffect "; inlineasm", "~{q0},~{q1},~{q2},~{q3},~{q4},~{q5},~{q6},~{q7},~{q8},~{q9},~{q10},~{q11},~{q12},~{q13},~{q14},~{q15},~{q16},~{q17},~{q18},~{q19},~{q20},~{q21},~{q22},~{q23},~{q24},~{q25},~{q26},~{q27},~{q28},~{q29},~{q30},~{q31},~{x0},~{x1},~{x2},~{x3},~{x4},~{x5},~{x6},~{x7},~{x8},~{x9},~{x10},~{x11},~{x12},~{x13},~{x14},~{x15},~{x16},~{x17},~{x18},~{x19},~{x20},~{x21},~{x22},~{x23},~{x24},~{x25},~{x26},~{x27},~{x28},~{fp},~{lr},~{sp},~{memory}"() nounwind
  store <4 x float> %x, ptr %p, align 16
  ret void
}
