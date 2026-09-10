; RUN: opt -passes=loop-unroll-and-jam -allow-unroll-and-jam -unroll-and-jam-count=4 -S < %s | FileCheck %s
;
; A constant trip count only clones the loop. Those counts stay counts.

; CHECK-NOT: Function Attrs: approxprofile
; CHECK: define void @jammed(

define void @jammed(ptr noalias nocapture %A, ptr noalias nocapture readonly %B) {
entry:
  br label %for.outer

for.outer:
  %i = phi i32 [ %i.next, %for.latch ], [ 0, %entry ]
  br label %for.inner

for.inner:
  %j = phi i32 [ 0, %for.outer ], [ %j.next, %for.inner ]
  %sum = phi i32 [ 0, %for.outer ], [ %add, %for.inner ]
  %b.ptr = getelementptr inbounds i32, ptr %B, i32 %j
  %b = load i32, ptr %b.ptr, align 4
  %add = add i32 %b, %sum
  %j.next = add nuw i32 %j, 1
  %inner.exit = icmp eq i32 %j.next, 8
  br i1 %inner.exit, label %for.latch, label %for.inner, !prof !0

for.latch:
  %sum.lcssa = phi i32 [ %add, %for.inner ]
  %a.ptr = getelementptr inbounds i32, ptr %A, i32 %i
  store i32 %sum.lcssa, ptr %a.ptr, align 4
  %i.next = add nuw i32 %i, 1
  %outer.exit = icmp eq i32 %i.next, 8
  br i1 %outer.exit, label %exit, label %for.outer, !prof !0

exit:
  ret void
}

!0 = !{!"branch_weights", i32 1, i32 7}
