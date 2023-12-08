; RUN: llc -O2  -mtriple=powerpc64-linux-gnu -mcpu=pwr8 < %s | FileCheck %s

define i64 @foo() {
entry:
  ret i64 -3617008641903833651

; CHECK: lis [[REG1:[0-9]+]], -12851
; CHECK: ori [[REG2:[0-9]+]], [[REG1]], 52685
; CHECK: rldimi 3, 3, 32, 0
}

