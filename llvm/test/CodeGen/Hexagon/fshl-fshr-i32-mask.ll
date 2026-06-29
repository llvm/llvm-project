; RUN: llc -mtriple=hexagon < %s | FileCheck %s

; Variable-amount 32-bit funnel shifts (and the rotates that lower to them)
; are realized as a 64-bit asl/lsr of the {hi,lo} combine, taking one word of
; the result. That identity only holds for a shift amount in [0,31], and
; llvm.fshl/fshr define the amount to be taken modulo 32. Hexagon's register
; variable shifts do NOT reduce the amount modulo the operand width -- they
; treat the low 7 bits as a signed amount in [-64,63] -- so the amount must be
; explicitly masked with #31 before the 64-bit shift. Without the mask, counts
; >= 32 or "negative" counts (e.g. -1 == 0xffffffff) are miscompiled.

; CHECK-LABEL: rotl_var:
; CHECK: r[[A0:[0-9]+]] = and(r1,#31)
; CHECK: = asl(r{{[0-9]+}}:{{[0-9]+}},r[[A0]])
define i32 @rotl_var(i32 %n, i32 %c) {
  %r = tail call i32 @llvm.fshl.i32(i32 %n, i32 %n, i32 %c)
  ret i32 %r
}

; CHECK-LABEL: rotr_var:
; CHECK: r[[A1:[0-9]+]] = and(r1,#31)
; CHECK: = lsr(r{{[0-9]+}}:{{[0-9]+}},r[[A1]])
define i32 @rotr_var(i32 %n, i32 %c) {
  %r = tail call i32 @llvm.fshr.i32(i32 %n, i32 %n, i32 %c)
  ret i32 %r
}

; CHECK-LABEL: fshl_var:
; CHECK: r[[A2:[0-9]+]] = and(r2,#31)
; CHECK: = asl(r{{[0-9]+}}:{{[0-9]+}},r[[A2]])
define i32 @fshl_var(i32 %a, i32 %b, i32 %c) {
  %r = tail call i32 @llvm.fshl.i32(i32 %a, i32 %b, i32 %c)
  ret i32 %r
}

; CHECK-LABEL: fshr_var:
; CHECK: r[[A3:[0-9]+]] = and(r2,#31)
; CHECK: = lsr(r{{[0-9]+}}:{{[0-9]+}},r[[A3]])
define i32 @fshr_var(i32 %a, i32 %b, i32 %c) {
  %r = tail call i32 @llvm.fshr.i32(i32 %a, i32 %b, i32 %c)
  ret i32 %r
}

declare i32 @llvm.fshl.i32(i32, i32, i32)
declare i32 @llvm.fshr.i32(i32, i32, i32)
