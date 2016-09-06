; RUN: opt -S -jump-threading < %s | FileCheck %s

; Check that based solely on static profile estimation we don't update the
; branch-weight metadata.  Even if the function has an entry frequency, a
; completely cold part of the CFG may be statically estimated.

; For example in the loop below, jump threading would update the weight of the
; loop-exiting branch to 0, drastically inflating the frequency of the loop
; (in the range of billions).
;
; This is the CFG of the loop.  There is no run-time profile info for edges
; inside the loop, so branch and block frequencies are estimated as shown:
;
;                 check_1 (16)
;             (8) /  |
;             eq_1   | (8)
;                 \  |
;                 check_2 (16)
;             (8) /  |
;             eq_2   | (8)
;                 \  |
;                 check_3 (16)
;             (1) /  |
;        (loop exit) | (15)
;                    |
;               (back edge)
;
; First we thread eq_1->check_2 to check_3.  Frequencies are updated to remove
; the frequency of eq_1 from check_2 and then the false edge leaving check_2
; (changed frequencies are highlighted with * *):
;
;                 check_1 (16)
;             (8) /  |
;            eq_1~   | (8)
;            /       |
;           /     check_2 (*8*)
;          /  (8) /  |
;          \  eq_2   | (*0*)
;           \     \  |
;            ` --- check_3 (16)
;             (1) /  |
;        (loop exit) | (15)
;                    |
;               (back edge)
;
; Next we thread eq_1->check_3 and eq_2->check_3 to check_1 as new back edges.
; Frequencies are updated to remove the frequency of eq_1 and eq_3 from
; check_3 and then the false edge leaving check_3 (changed frequencies are
; highlighted with * *):
;
;                 check_1 (16)
;             (8) /  |
;            eq_1~   | (8)
;            /       |
;           /     check_2 (*8*)
;          /  (8) /  |
;         /-- eq_2~  | (*0*)
; (back edge)        |
;                 check_3 (*0*)
;           (*0*) /  |
;        (loop exit) | (*0*)
;                    |
;               (back edge)
;
; As a result, the loop exit edge ends up with 0 frequency which in turn makes
; the loop header to have maximum frequency.

declare void @bar()

define void @foo(i32 *%p, i32 %n) !prof !0 {
entry:
  %enter_loop = icmp eq i32 %n, 0
  br i1 %enter_loop, label %exit, label %check_1, !prof !1
; CHECK: br i1 %enter_loop, label %exit, label %check_1, !prof !1

check_1:
  %v = load i32, i32* %p
  %cond1 = icmp eq i32 %v, 1
  br i1 %cond1, label %eq_1, label %check_2
; No metadata:
; CHECK:   br i1 %cond1, label %check_2.thread, label %check_2{{$}}

eq_1:
  call void @bar()
  br label %check_2
; Verify the new backedge:
; CHECK: check_2.thread:
; CHECK-NEXT: call void @bar()
; CHECK-NEXT: br label %check_1

check_2:
  %cond2 = icmp eq i32 %v, 2
  br i1 %cond2, label %eq_2, label %check_3
; No metadata:
; CHECK: br i1 %cond2, label %eq_2, label %check_3{{$}}

eq_2:
  call void @bar()
  br label %check_3
; Verify the new backedge:
; CHECK: eq_2:
; CHECK-NEXT: call void @bar()
; CHECK-NEXT: br label %check_1

check_3:
  %condE = icmp eq i32 %v, 3
  br i1 %condE, label %exit, label %check_1
; No metadata:
; CHECK: br i1 %condE, label %exit, label %check_1{{$}}

exit:
  ret void
}

!0 = !{!"function_entry_count", i64 120}
; CHECK-NOT: branch_weights
!1 = !{!"branch_weights", i32 119, i32 1}
; CHECK: !1 = !{!"branch_weights", i32 119, i32 1}
; CHECK-NOT: branch_weights
