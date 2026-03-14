; RUN: llc -mtriple=thumb-eabi --stop-after=peephole-opt -mcpu=arm1156t2-s -mattr=+thumb2 %s -o - | FileCheck %s
define i32 @t2_const_var2_1_ok_2(i32 %lhs) {
; CHECK: [[R0:%0|%[1-9][0-9]*]]:gprnopc = COPY $r0
; CHECK-NEXT: [[R1:%0|%[1-9][0-9]*]]:rgpr = t2ADDri [[R0]], 11206656
; CHECK-NEXT: [[R2:%0|%[1-9][0-9]*]]:rgpr = t2ADDri killed [[R1]], 187
; CHECK-NEXT: $r0 = COPY [[R2]]
  %ret = add i32 %lhs, 11206843 ; 0x00ab00bb
  ret i32 %ret
}

