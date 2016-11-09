; RUN: opt < %s -jump-threading -instcombine -simplifycfg  -S | FileCheck %s

; The three selects are jump-threaded so that instcombine can optimize, and
; simplifycfg should turn the result into a single select.
define i32 @f(i32 %a, i32 %b) {
; CHECK: select
; CHECK-NOT: select
entry:
  %0 = and i32 %a, 1
  %1 = and i32 %b, 1
  %xor = xor i32 %1, %a
  %shr32 = lshr i32 %a, 1
  %cmp10 = icmp eq i32 %xor, 1
  %2 = xor i32 %b, 12345
  %b.addr.1 = select i1 %cmp10, i32 %2, i32 %b
  %shr1633 = lshr i32 %b.addr.1, 1
  %3 = or i32 %shr1633, 54321
  %b.addr.2 = select i1 %cmp10, i32 %3, i32 %shr1633
  %shr1634 = lshr i32 %b.addr.2, 2
  %4 = or i32 %shr1634, 54320
  %b.addr.3 = select i1 %cmp10, i32 %4, i32 %shr1634
  ret i32 %b.addr.3
}

; Case where the condition is not only used as condition but also as the
; true or false value in at least one of the selects.
define i1 @g(i32 %a, i32 %b) {
; CHECK: select
; CHECK-NOT: select
entry:
  %0 = and i32 %a, 1
  %1 = and i32 %b, 1
  %xor = xor i32 %1, %a
  %shr32 = lshr i32 %a, 1
  %cmp10 = icmp eq i32 %xor, 1
  %2 = xor i32 %b, 12345
  %b.addr.1 = select i1 %cmp10, i32 %2, i32 %b
  %shr1633 = lshr i32 %b.addr.1, 1
  %3 = or i32 %shr1633, 54321
  %b.addr.2 = select i1 %cmp10, i32 %3, i32 %shr1633
  %shr1634 = lshr i32 %b.addr.2, 2
  %4 = icmp eq i32 %shr1634, 54320
  %b.addr.3 = select i1 %cmp10, i1 %4, i1 %cmp10
  ret i1 %b.addr.3
}
