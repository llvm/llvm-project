; RUN: llc -O2 -mtriple=x86_64-unknown-linux-gnu < %s | FileCheck %s
;
; Do not rotate a single-successor increment block in front of the latch when
; a 2-way predecessor already has the latch as its other successor:
;
;   header --> check_nl --> latch --> header
;        \        |          ^
;         \       v          |
;          ----> inc --------
;
; Rotating inc to the loop top inverts the likely fallthrough of check_nl
; (cmpb $10 / je to inc) and increases branch misses. See GH218248.

; CHECK-LABEL: skip_sep:
; CHECK:       %check_nl
; CHECK:       cmpb $10, %cl
; CHECK-NEXT:  jne

define ptr @skip_sep(ptr %p, ptr %end) {
entry:
  br label %header

header:
  %q = phi ptr [ %p, %entry ], [ %q.next, %latch ]
  %c = load i8, ptr %q
  %is_comma = icmp eq i8 %c, 44
  br i1 %is_comma, label %inc, label %check_nl

check_nl:
  %is_nl = icmp eq i8 %c, 10
  br i1 %is_nl, label %inc, label %latch

inc:
  %q.inc = getelementptr i8, ptr %q, i64 1
  br label %latch

latch:
  %q.next = phi ptr [ %q, %check_nl ], [ %q.inc, %inc ]
  %done = icmp uge ptr %q.next, %end
  br i1 %done, label %exit, label %header

exit:
  ret ptr %q.next
}
