; RUN: opt -S -irce -irce-print-changed-loops=true < %s | FileCheck %s
; RUN: opt -S -passes='require<branch-prob>,irce' -irce-print-changed-loops=true < %s | FileCheck %s

; CHECK-NOT: irce

define void @bad_loop_structure_increasing(i64 %iv.start) {
entry:
  br label %for.body

for.body:
  %indvars.iv = phi i64 [ %iv.start, %entry ], [ %indvars.iv.next, %for.inc ]
  %cmp = icmp ult i64 %indvars.iv, 100
  br i1 %cmp, label %switch.lookup, label %for.inc

switch.lookup:
  br label %for.inc

for.inc:
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %cmp55 = icmp slt i64 %indvars.iv.next, 11
  br i1 %cmp55, label %for.body, label %for.end

for.end:
  ret void
}

define void @bad_loop_structure_decreasing(i64 %iv.start) {
entry:
  br label %for.body

for.body:
  %indvars.iv = phi i64 [ %iv.start, %entry ], [ %indvars.iv.next, %for.inc ]
  %cmp = icmp ult i64 %indvars.iv, 100
  br i1 %cmp, label %switch.lookup, label %for.inc

switch.lookup:
  br label %for.inc

for.inc:
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, -1
  %cmp55 = icmp sgt i64 %indvars.iv.next, 11
  br i1 %cmp55, label %for.body, label %for.end

for.end:
  ret void
}
