; RUN: llc < %s -march=nios2 2>&1 | FileCheck %s
; RUN: llc < %s -march=nios2 -target-abi=nios2r2 2>&1 | FileCheck %s

define i32 @add_reg(i32 %a, i32 %b) nounwind {
entry:
; CHECK: add_reg:
; CHECK:   add r2, r4, r5
  %c = add i32 %a, %b
  ret i32 %c
}

define i32 @sub_reg(i32 %a, i32 %b) nounwind {
entry:
; CHECK: sub_reg:
; CHECK:   sub r2, r4, r5
  %c = sub i32 %a, %b
  ret i32 %c
}

