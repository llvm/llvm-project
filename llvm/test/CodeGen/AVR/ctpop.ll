; RUN: llc < %s -march=avr | FileCheck %s

define i8 @count_population(i8) unnamed_addr {
entry-block:
  %1 = tail call i8 @llvm.ctpop.i8(i8 %0)
  ret i8 %1
}

declare i8 @llvm.ctpop.i8(i8)

; CHECK-LABEL: count_population:
; CHECK: mov    [[SCRATCH:r[0-9]+]], [[RESULT:r[0-9]+]]
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
; CHECK: lsr    {{.*}}[[SCRATCH]]
; CHECK: lsr    {{.*}}[[SCRATCH]]
; CHECK: lsr    {{.*}}[[SCRATCH]]
; CHECK: lsr    {{.*}}[[SCRATCH]]
; CHECK: add    {{.*}}[[SCRATCH]], {{.*}}[[RESULT]]
; CHECK: andi   {{.*}}[[SCRATCH]], 15
; CHECK: mov    {{.*}}[[RESULT]], {{.*}}[[SCRATCH]]
; CHECK: ret
