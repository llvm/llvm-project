; RUN: llc -mtriple=hexagon < %s | FileCheck %s

; Verify that SelectSHL handles (shl (mul val, const), const) with negative
; multiply constants. The left shift of the negative constant must be done in
; unsigned arithmetic to avoid undefined behavior.

; The multiply constant 0xAAAAAAAB shifted left by 1 does not fit in a signed
; 9-bit immediate, so this should not be folded into M2_mpysmi.
define i32 @shl_mul_neg_large(i32 %x) #0 {
; CHECK-LABEL: shl_mul_neg_large:
; CHECK: +mpyi(r0,##1431655766)
; CHECK: jumpr r31
entry:
  %m = mul i32 %x, u0xAAAAAAAB
  %s = shl i32 %m, 1
  ret i32 %s
}

; A small negative multiply constant (-3) shifted left by 1 gives -6, which
; fits in a signed 9-bit immediate. This should be folded into M2_mpysmi.
define i32 @shl_mul_neg_small(i32 %x) #0 {
; CHECK-LABEL: shl_mul_neg_small:
; CHECK: -mpyi(r0,#6)
; CHECK: jumpr r31
entry:
  %m = mul i32 %x, -3
  %s = shl i32 %m, 1
  ret i32 %s
}

attributes #0 = { nounwind "target-cpu"="hexagonv60" }
