; RUN: llc < %s -march=avr | FileCheck %s

define i8 @count_trailing_zeros(i8) unnamed_addr {
entry-block:
  %1 = tail call i8 @llvm.cttz.i8(i8 %0)
  ret i8 %1
}

declare i8 @llvm.cttz.i8(i8)

; CHECK-LABEL: count_trailing_zeros:
; CHECK: cpi    [[RESULT:r[0-9]+]], 0
; CHECK: breq   [[END_BB:.LBB[0-9]+_[0-9]+]]
; CHECK: mov    [[SCRATCH:r[0-9]+]], {{.*}}[[RESULT]]
; CHECK: dec    {{.*}}[[SCRATCH]]
; CHECK: com    {{.*}}[[RESULT]]
; CHECK: and    {{.*}}[[RESULT]], {{.*}}[[SCRATCH]]
; CHECK: mov    {{.*}}[[SCRATCH]], {{.*}}[[RESULT]]
; CHECK: lsr    {{.*}}[[SCRATCH]]
; CHECK: andi   {{.*}}[[SCRATCH]], 85
; CHECK: sub    {{.*}}[[RESULT]], {{.*}}[[SCRATCH]]
; CHECK: mov    {{.*}}[[SCRATCH]], {{.*}}[[RESULT]]
; CHECK: andi   {{.*}}[[SCRATCH]], 51
; CHECK: lsr    {{.*}}[[RESULT]]
; CHECK: lsr    {{.*}}[[RESULT]]
; CHECK: andi   {{.*}}[[RESULT]], 51
; CHECK: add    {{.*}}[[RESULT]], {{.*}}[[SCRATCH]]
; CHECK: mov    {{.*}}[[SCRATCH]], {{.*}}[[RESULT]]
; CHECK: swap   {{.*}}[[SCRATCH]]
; CHECK: add    {{.*}}[[SCRATCH]], {{.*}}[[RESULT]]
; CHECK: mov    {{.*}}[[RESULT]], {{.*}}[[SCRATCH]]
; CHECK: andi   {{.*}}[[RESULT]], 15
; CHECK: ret
; CHECK: [[END_BB]]:
; CHECK: ldi    {{.*}}[[RESULT]], 8
; CHECK: ret
