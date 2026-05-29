; The bfloat16-cast-decompose pass also handles uitofp (unsigned int -> bf16),
; not just sitofp covered by bfloat16-cast-decompose.ll. Pin the same shape:
; uitofp -> float intermediate -> bit shuffling -> bitcast to bfloat.
; RUN: llc -mtriple=air %s -o - | FileCheck %s

; CHECK-LABEL: define bfloat @to_bf16_unsigned
; CHECK: uitofp i32 %x to float
; CHECK: bitcast float {{.*}} to i32
; CHECK: lshr i32 {{.*}}, 16
; CHECK: trunc i32 {{.*}} to i16
; CHECK: bitcast i16 {{.*}} to bfloat

define bfloat @to_bf16_unsigned(i32 %x) {
  %r = uitofp i32 %x to bfloat
  ret bfloat %r
}
