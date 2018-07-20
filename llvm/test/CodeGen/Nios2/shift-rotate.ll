; RUN: llc < %s -march=nios2 2>&1 | FileCheck %s
; RUN: llc < %s -march=nios2 -target-abi=nios2r2 2>&1 | FileCheck %s

define i32 @sll_reg(i32 %a, i32 %b) nounwind {
entry:
; CHECK: sll_reg:
; CHECK:   sll r2, r4, r5
  %c = shl i32 %a, %b
  ret i32 %c
}

define i32 @srl_reg(i32 %a, i32 %b) nounwind {
entry:
; CHECK: srl_reg:
; CHECK:   srl r2, r4, r5
  %c = lshr i32 %a, %b
  ret i32 %c
}

define i32 @sra_reg(i32 %a, i32 %b) nounwind {
entry:
; CHECK: sra_reg:
; CHECK:   sra r2, r4, r5
  %c = ashr i32 %a, %b
  ret i32 %c
}
