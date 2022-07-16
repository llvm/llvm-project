; Test floating point arithmetic.
;
; RUN: llc < %s -mtriple=m88k-openbsd -mcpu=mc88100 -O0 | FileCheck --check-prefix=CHECK %s
; RUN: llc < %s -mtriple=m88k-openbsd -mcpu=mc88110 -O0 | FileCheck --check-prefix=CHECK %s

define i64 @udiv64(i64 %a, i64 %b) {
; CHECK-LABEL: udiv64:
; CHECK: bsr __udivdi3
; CHECK: jmp %r1
  %quot = udiv i64 %a, %b
  ret i64 %quot
}

define i64 @sdiv64(i64 %a, i64 %b) {
; CHECK-LABEL: sdiv64:
; CHECK: bsr __divdi3
; CHECK: jmp %r1
  %quot = sdiv i64 %a, %b
  ret i64 %quot
}

define i64 @urem64(i64 %a, i64 %b) {
; CHECK-LABEL: urem64:
; CHECK: bsr __umoddi3
; CHECK: jmp %r1
  %quot = urem i64 %a, %b
  ret i64 %quot
}

define i64 @srem64(i64 %a, i64 %b) {
; CHECK-LABEL: srem64:
; CHECK: bsr __moddi3
; CHECK: jmp %r1
  %quot = srem i64 %a, %b
  ret i64 %quot
}
