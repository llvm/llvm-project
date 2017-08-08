; RUN: opt -verify-loop-info -irce -S < %s

; These test cases don't check the correctness of the transform, but
; that -irce does not crash in the presence of certain things in
; the IR:

define void @mismatched_types_1() {
; In this test case, the safe range for the only range check in the
; loop is of type [i32, i32) while the backedge taken count is of type
; i64.

; CHECK-LABEL: @mismatched_types_1(
entry:
  br label %for.body

for.body:
  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %for.inc ]
  %0 = trunc i64 %indvars.iv to i32
  %1 = icmp ult i32 %0, 7
  br i1 %1, label %switch.lookup, label %for.inc

switch.lookup:
  br label %for.inc

for.inc:
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %cmp55 = icmp slt i64 %indvars.iv.next, 11
  br i1 %cmp55, label %for.body, label %for.end

for.end:
  unreachable
}

define void @mismatched_types_2() {
; In this test case, there are two range check in the loop, one with a
; safe range of type [i32, i32) and one with a safe range of type
; [i64, i64).

; CHECK-LABEL: @mismatched_types_2(
entry:
  br label %for.body.a

for.body.a:
  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %for.inc ]
  %cond.a = icmp ult i64 %indvars.iv, 7
  br i1 %cond.a, label %switch.lookup.a, label %for.body.b

switch.lookup.a:
  br label %for.body.b

for.body.b:
  %truncated = trunc i64 %indvars.iv to i32
  %cond.b = icmp ult i32 %truncated, 7
  br i1 %cond.b, label %switch.lookup.b, label %for.inc

switch.lookup.b:
  br label %for.inc

for.inc:
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %cmp55 = icmp slt i64 %indvars.iv.next, 11
  br i1 %cmp55, label %for.body.a, label %for.end

for.end:
  unreachable
}
