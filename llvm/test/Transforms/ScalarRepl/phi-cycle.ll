; RUN: opt -S -scalarrepl-ssa < %s | FileCheck %s
; rdar://10589171

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

%struct.foo = type { i32, i32 }

@.str = private unnamed_addr constant [6 x i8] c"x=%d\0A\00", align 1

define i32 @main(i32 %argc, i8** nocapture %argv) nounwind uwtable {
entry:
  %f = alloca %struct.foo, align 4
  %x.i = getelementptr inbounds %struct.foo* %f, i64 0, i32 0
  store i32 1, i32* %x.i, align 4
  %y.i = getelementptr inbounds %struct.foo* %f, i64 0, i32 1
  br label %while.cond.i

; CHECK: while.cond.i:
; CHECK-NEXT: %tmp = phi i32 [ 1, %entry ], [ %tmp2, %while.cond.backedge.i ]
; CHECK-NEXT: %pos.0.i = phi i32 [ 1, %entry ], [ %xtmp.i, %while.cond.backedge.i ]
; CHECK-NEXT: %left.0.i = phi i32 [ 1, %entry ], [ %dec.i, %while.cond.backedge.i ]
; CHECK-NOT: phi
while.cond.i:                                     ; preds = %while.cond.backedge.i, %entry
  %tmp = phi i32 [ 1, %entry ], [ %tmp2, %while.cond.backedge.i ]
  %pos.0.i = phi i32 [ 1, %entry ], [ %xtmp.i, %while.cond.backedge.i ]
  %left.0.i = phi i32 [ 1, %entry ], [ %dec.i, %while.cond.backedge.i ]
  %cmp.i = icmp sgt i32 %left.0.i, 0
  br i1 %cmp.i, label %while.body.i, label %while.cond.i.func.exit_crit_edge

while.cond.i.func.exit_crit_edge:                 ; preds = %while.cond.i
  br label %func.exit

while.body.i:                                     ; preds = %while.cond.i
  %dec.i = add nsw i32 %left.0.i, -1
  switch i32 1, label %while.body.i.func.exit_crit_edge [
    i32 0, label %while.cond.backedge.i
    i32 1, label %sw.bb.i
  ]

while.body.i.func.exit_crit_edge:                 ; preds = %while.body.i
  br label %func.exit

sw.bb.i:                                          ; preds = %while.body.i
  %cmp2.i = icmp eq i32 %tmp, 1
  br i1 %cmp2.i, label %if.then.i, label %if.end.i

if.then.i:                                        ; preds = %sw.bb.i
  store i32 %pos.0.i, i32* %x.i, align 4
  br label %if.end.i

; CHECK: if.end.i:
; CHECK-NEXT: %tmp1 = phi i32 [ %pos.0.i, %if.then.i ], [ %tmp, %sw.bb.i ]
; CHECK-NOT: phi
if.end.i:                                         ; preds = %if.then.i, %sw.bb.i
  %tmp1 = phi i32 [ %pos.0.i, %if.then.i ], [ %tmp, %sw.bb.i ]
  store i32 %tmp1, i32* %y.i, align 4
  br label %while.cond.backedge.i

; CHECK: while.cond.backedge.i:
; CHECK-NEXT: %tmp2 = phi i32 [ %tmp1, %if.end.i ], [ %tmp, %while.body.i ]
; CHECK-NOT: phi
while.cond.backedge.i:                            ; preds = %if.end.i, %while.body.i
  %tmp2 = phi i32 [ %tmp1, %if.end.i ], [ %tmp, %while.body.i ]
  %xtmp.i = add i32 %pos.0.i, 1
  br label %while.cond.i

; CHECK: func.exit:
; CHECK-NOT: load
; CHECK: %call = call i32 (i8*, ...)* @printf(i8* getelementptr inbounds ([6 x i8]* @.str, i64 0, i64 0), i32 %tmp) nounwind
func.exit:                                        ; preds = %while.body.i.func.exit_crit_edge, %while.cond.i.func.exit_crit_edge
  %tmp3 = load i32* %x.i, align 4
  %call = call i32 (i8*, ...)* @printf(i8* getelementptr inbounds ([6 x i8]* @.str, i64 0, i64 0), i32 %tmp3) nounwind
  ret i32 0
}

declare i32 @printf(i8* nocapture, ...) nounwind
