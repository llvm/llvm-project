; RUN: llc < %s -march=nios2 2>&1 | FileCheck %s
; RUN: llc < %s -march=nios2 -target-abi=nios2r2 2>&1 | FileCheck %s

define i32 @mul_reg(i32 %a, i32 %b) nounwind {
entry:
; CHECK: mul_reg:
; CHECK:   mul r2, r4, r5
  %c = mul i32 %a, %b
  ret i32 %c
}

define i32 @div_signed(i32 %a, i32 %b) nounwind {
entry:
; CHECK: div_signed:
; CHECK:   div r2, r4, r5
  %c = sdiv i32 %a, %b
  ret i32 %c
}

define i32 @div_unsigned(i32 %a, i32 %b) nounwind {
entry:
; CHECK: div_unsigned:
; CHECK:   divu r2, r4, r5
  %c = udiv i32 %a, %b
  ret i32 %c
}

