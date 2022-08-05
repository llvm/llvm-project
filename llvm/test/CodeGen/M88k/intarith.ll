; Test floating point arithmetic.
;
; RUN: llc < %s -mtriple=m88k-openbsd -mcpu=mc88100 -m88k-enable-delay-slot-filler=false | FileCheck --check-prefixes=CHECK,MC88100 %s
; RUN: llc < %s -mtriple=m88k-openbsd -mcpu=mc88110 -m88k-enable-delay-slot-filler=false | FileCheck --check-prefixes=CHECK,MC88110 %s

define i64 @udiv64(i64 %a, i64 %b) {
; CHECK-LABEL: udiv64:
; CHECK: bsr __udivdi3
; CHECK: jmp %r1
  %quot = udiv i64 %a, %b
  ret i64 %quot
}

define i64 @udiv64with32(i64 %a, i32 %b) {
; CHECK-LABEL: udiv64with32:
; MC88100: bsr __udivdi3
; MC88110: divu.d %r2, %r2, %r4
; CHECK: jmp %r1
  %conv = zext i32 %b to i64
  %quot = udiv i64 %a, %conv
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

define i64 @mul64(i64 %a, i64 %b) {
; CHECK-LABEL: mul64:
; CHECK: bsr __muldi3
; CHECK: jmp %r1
  %mult = mul i64 %a, %b
  ret i64 %mult
}

define i64 @mul32to64(i32 %a, i32 %b) {
; CHECK-LABEL: mul32to64:
; MC88100: bsr __muldi3
; MC88110: mulu.d %r2, %r2, %r3
; CHECK: jmp %r1
  %conva = zext i32 %a to i64
  %convb = zext i32 %b to i64
  %mult = mul i64 %conva, %convb
  ret i64 %mult
}
