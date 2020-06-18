; RUN: llc < %s -march=avr | FileCheck %s

define i8 @count_leading_zeros(i8) unnamed_addr {
entry-block:
  %1 = tail call i8 @llvm.ctlz.i8(i8 %0)
  ret i8 %1
}

declare i8 @llvm.ctlz.i8(i8)

; CHECK-LABEL: count_leading_zeros:
; CHECK: cpi    [[RESULT:r[0-9]+]], 0
; CHECK: brne   .LBB0_1
; CHECK: rjmp   .LBB0_2
; CHECK: mov    [[SCRATCH:r[0-9]+]], {{.*}}[[RESULT]]
; CHECK: lsr    {{.*}}[[SCRATCH]]
; CHECK: or     {{.*}}[[SCRATCH]], {{.*}}[[RESULT]]
; CHECK: mov    {{.*}}[[RESULT]], {{.*}}[[SCRATCH]]
; CHECK: lsr    {{.*}}[[RESULT]]
; CHECK: lsr    {{.*}}[[RESULT]]
; CHECK: or     {{.*}}[[RESULT]], {{.*}}[[SCRATCH]]
; CHECK: mov    {{.*}}[[SCRATCH]], {{.*}}[[RESULT]]
; CHECK: lsr    {{.*}}[[SCRATCH]]
; CHECK: lsr    {{.*}}[[SCRATCH]]
; CHECK: lsr    {{.*}}[[SCRATCH]]
; CHECK: lsr    {{.*}}[[SCRATCH]]
; CHECK: or     {{.*}}[[SCRATCH]], {{.*}}[[RESULT]]
; CHECK: com    {{.*}}[[SCRATCH]]
; CHECK: mov    {{.*}}[[RESULT]], {{.*}}[[SCRATCH]]
; CHECK: lsr    {{.*}}[[RESULT]]
; CHECK: andi   {{.*}}[[RESULT]], 85
; CHECK: sub    {{.*}}[[SCRATCH]], {{.*}}[[RESULT]]
; CHECK: mov    {{.*}}[[RESULT]], {{.*}}[[SCRATCH]]
; CHECK: andi   {{.*}}[[RESULT]], 51
; CHECK: lsr    {{.*}}[[SCRATCH]]
; CHECK: lsr    {{.*}}[[SCRATCH]]
; CHECK: andi   {{.*}}[[SCRATCH]], 51
; CHECK: add    {{.*}}[[SCRATCH]], {{.*}}[[RESULT]]
; CHECK: mov    {{.*}}[[RESULT]], {{.*}}[[SCRATCH]]
; CHECK: lsr    {{.*}}[[RESULT]]
; CHECK: lsr    {{.*}}[[RESULT]]
; CHECK: lsr    {{.*}}[[RESULT]]
; CHECK: lsr    {{.*}}[[RESULT]]
; CHECK: add    {{.*}}[[RESULT]], {{.*}}[[SCRATCH]]
; CHECK: andi   {{.*}}[[RESULT]], 15
; CHECK: ret
; CHECK: LBB0_2:
; CHECK: ldi    {{.*}}[[RESULT]], 8
; CHECK: ret
