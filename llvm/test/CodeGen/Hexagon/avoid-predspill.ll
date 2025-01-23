; RUN: llc -mtriple=hexagon -O2 < %s | FileCheck %s
;
; This checks that predicate registers are moved to GPRs instead of spilling
; where possible.

; CHECK: p0 =
; CHECK-NOT: memw(r29

define i32 @f(i32 %a, i32 %b, i32 %c, i32 %d, i32 %e) {
entry:
  %cmp = icmp eq i32 %a, 1
  %cmp1 = icmp eq i32 %b, 2
  %or.cond = and i1 %cmp, %cmp1
  %cmp3 = icmp eq i32 %c, 3
  %or.cond30 = and i1 %or.cond, %cmp3
  %cmp5 = icmp eq i32 %d, 4
  %or.cond31 = and i1 %or.cond30, %cmp5
  %cmp7 = icmp eq i32 %e, 5
  %or.cond32 = and i1 %or.cond31, %cmp7
  %ret.0 = zext i1 %or.cond32 to i32
  %cmp8 = icmp eq i32 %a, 3
  %cmp10 = icmp eq i32 %b, 4
  %or.cond33 = and i1 %cmp8, %cmp10
  %cmp12 = icmp eq i32 %c, 5
  %or.cond34 = and i1 %or.cond33, %cmp12
  %cmp14 = icmp eq i32 %d, 6
  %or.cond35 = and i1 %or.cond34, %cmp14
  %cmp16 = icmp eq i32 %e, 7
  %or.cond36 = and i1 %or.cond35, %cmp16
  %ret.1 = select i1 %or.cond36, i32 2, i32 %ret.0
  %cmp21 = icmp eq i32 %b, 8
  %or.cond37 = and i1 %cmp, %cmp21
  %cmp23 = icmp eq i32 %c, 2
  %or.cond38 = and i1 %or.cond37, %cmp23
  %cmp25 = icmp eq i32 %d, 1
  %or.cond39 = and i1 %or.cond38, %cmp25
  %cmp27 = icmp eq i32 %e, 3
  %or.cond40 = and i1 %or.cond39, %cmp27
  %ret.2 = select i1 %or.cond40, i32 3, i32 %ret.1
  ret i32 %ret.2
}

