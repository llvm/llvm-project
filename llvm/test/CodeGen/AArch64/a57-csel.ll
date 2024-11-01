; RUN: llc -mtriple=aarch64-none-linux-gnu < %s -mcpu=cortex-a57 -aarch64-enable-early-ifcvt=false | FileCheck %s

; Check that the select isn't expanded into a branch sequence
; when the icmp's first operand %x0 is from load.
define i64 @f(i64 %a, i64 %b, ptr %c, i64 %d, i64 %e) {
  ; CHECK: csel
  %x0 = load i64, ptr %c
  %x1 = icmp eq i64 %x0, 0
  %x2 = select i1 %x1, i64 %a, i64 %b
  %x3 = add i64 %x2, %d
  ret i64 %x3
}
