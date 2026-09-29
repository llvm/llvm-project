; RUN: llc -o - %s | FileCheck %s
target triple = "arm-unknown-unknown"

; select with and i1/or i1 condition should be implemented as a series of 2
; cmovs, not by producing two conditions and using and on them.

define i32 @select_and(i32 %a0, i32 %a1, i32 %a2, i32 %a3, i32 %a4, i32 %a5) {
; CHECK-LABEL: select_and
; CHECK-NOT: tst
; CHECK-NOT: movne
; CHECK: mov{{lo|hs}}
; CHECK: mov{{lo|hs}}
  %cmp0 = icmp ult i32 %a0, %a1
  %cmp1 = icmp ult i32 %a2, %a3
  %and = and i1 %cmp0, %cmp1
  %res = select i1 %and, i32 %a4, i32 %a5
  ret i32 %res
}

define i32 @select_or(i32 %a0, i32 %a1, i32 %a2, i32 %a3, i32 %a4, i32 %a5) {
; select with and i1 condition should be implemented as a series of 2 cmovs, not
; by producing two conditions and using and on them.
; CHECK-LABEL: select_or
; CHECK-NOT: orss
; CHECK-NOT: tst
; CHECK: mov{{lo|hs}}
; CHECK: mov{{lo|hs}}
  %cmp0 = icmp ult i32 %a0, %a1
  %cmp1 = icmp ult i32 %a2, %a3
  %and = or i1 %cmp0, %cmp1
  %res = select i1 %and, i32 %a4, i32 %a5
  ret i32 %res
}

; If one of the conditions is materialized as a 0/1 value anyway, then the
; sequence of 2 cmovs should not be used.

@var32 = global i32 0
define i32 @select_noopt(i32 %a0, i32 %a1, i32 %a2, i32 %a3, i32 %a4) {
; CHECK-LABEL: select_noopt
; CHECK: orrs
; CHECK: movne
  %cmp0 = icmp ult i32 %a0, %a1
  %cmp1 = icmp ult i32 %a1, %a2
  %or = or i1 %cmp0, %cmp1
  %zero_one = zext i1 %or to i32
  store volatile i32 %zero_one, ptr @var32
  %res = select i1 %or, i32 %a3, i32 %a4
  ret i32 %res
}
