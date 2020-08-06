; REQUIRES: asserts

; RUN: opt -newgvn -S %s | FileCheck %s

; XFAIL: *

; TODO: Currently NewGVN crashes on the function below, see PR42422.

define void @d() {
entry:
  br label %for.cond

for.cond:                                         ; preds = %cleanup20, %entry
  br label %for.cond1

for.cond1:                                        ; preds = %for.inc17, %for.cond
  %0 = phi i32 [ %inc18, %for.inc17 ], [ 0, %for.cond ]
  %cmp = icmp sle i32 %0, 1
  br i1 %cmp, label %for.body, label %for.end19

for.body:                                         ; preds = %for.cond1
  br i1 undef, label %for.body3, label %for.body.for.cond4_crit_edge

for.body.for.cond4_crit_edge:                     ; preds = %for.body
  br label %for.cond4

for.body3:                                        ; preds = %for.body
  br label %cleanup14

for.cond4:                                        ; preds = %cleanup, %for.body.for.cond4_crit_edge
  br i1 undef, label %if.then, label %if.end

if.then:                                          ; preds = %for.cond4
  br label %cleanup

if.end:                                           ; preds = %for.cond4
  br label %for.cond6

for.cond6:                                        ; preds = %for.inc, %if.end
  %1 = phi i64 [ %inc, %for.inc ], [ 0, %if.end ]
  %cmp7 = icmp sle i64 %1, 1
  br i1 %cmp7, label %for.inc, label %for.end9

for.inc:                                          ; preds = %for.cond6
  %inc = add nsw i64 %1, 1
  br label %for.cond6

for.end9:                                         ; preds = %for.cond6
  br i1 true, label %if.then11, label %if.end12

if.then11:                                        ; preds = %for.end9
  br label %cleanup

if.end12:                                         ; preds = %for.end9
  br label %cleanup

cleanup:                                          ; preds = %if.end12, %if.then11, %if.then
  %cleanup.dest = phi i32 [ undef, %if.end12 ], [ 1, %if.then11 ], [ 9, %if.then ]
  switch i32 %cleanup.dest, label %cleanup14 [
    i32 0, label %for.cond4
    i32 9, label %for.end13
  ]

for.end13:                                        ; preds = %cleanup
  br label %cleanup14

cleanup14:                                        ; preds = %for.end13, %cleanup, %for.body3
  %cleanup.dest15 = phi i32 [ 0, %for.end13 ], [ %cleanup.dest, %cleanup ], [ 1, %for.body3 ]
  %cond1 = icmp eq i32 %cleanup.dest15, 0
  br i1 %cond1, label %for.inc17, label %cleanup20

for.inc17:                                        ; preds = %cleanup14
  %inc18 = add nsw i32 %0, 1
  br label %for.cond1

for.end19:                                        ; preds = %for.cond1
  br label %cleanup20

cleanup20:                                        ; preds = %for.end19, %cleanup14
  %cleanup.dest21 = phi i32 [ %cleanup.dest15, %cleanup14 ], [ 0, %for.end19 ]
  %cond = icmp eq i32 %cleanup.dest21, 0
  br i1 %cond, label %for.cond, label %cleanup23

cleanup23:                                        ; preds = %cleanup20
  ret void
}
