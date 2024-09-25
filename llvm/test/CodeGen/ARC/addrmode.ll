; RUN: llc -march=arc < %s | FileCheck %s

; CHECK-LABEL: copy
; CHECK-NOT: add
define void @copy(ptr inreg nocapture %p, ptr inreg nocapture readonly %q) {
entry:
  br label %while.cond

while.cond:                                       ; preds = %while.cond, %entry
  %p.addr.0 = phi ptr [ %p, %entry ], [ %incdec.ptr1, %while.cond ]
  %q.addr.0 = phi ptr [ %q, %entry ], [ %incdec.ptr, %while.cond ]
  %incdec.ptr = getelementptr inbounds i8, ptr %q.addr.0, i32 1
  %0 = load i8, ptr %q.addr.0, align 1
  %incdec.ptr1 = getelementptr inbounds i8, ptr %p.addr.0, i32 1
  store i8 %0, ptr %p.addr.0, align 1
  %tobool = icmp eq i8 %0, 0
  br i1 %tobool, label %while.end, label %while.cond

while.end:                                        ; preds = %while.cond
  ret void
}


%struct._llist = type { ptr, ptr, i32 }

; CHECK-LABEL: neg1
; CHECK-NOT:   std.ab
define void @neg1(ptr inreg nocapture %a, ptr inreg nocapture readonly %b, i32 inreg %n) {
entry:
  %cmp6 = icmp sgt i32 %n, 0
  br i1 %cmp6, label %for.body, label %for.cond.cleanup

for.cond.cleanup:
  ret void

for.body:
  %i.07 = phi i32 [ %inc, %for.body ], [ 0, %entry ]
  %arrayidx = getelementptr inbounds i8, ptr %b, i32 %i.07
  %0 = load i8, ptr %arrayidx, align 1
  %mul = mul nuw nsw i32 %i.07, 257
  %arrayidx1 = getelementptr inbounds i8, ptr %a, i32 %mul
  store i8 %0, ptr %arrayidx1, align 1
  %inc = add nuw nsw i32 %i.07, 1
  %exitcond = icmp eq i32 %inc, %n
  br i1 %exitcond, label %for.cond.cleanup, label %for.body
}

; CHECK-LABEL: neg2
; CHECK-NOT:   st.ab
define void @neg2(ptr inreg %a, i32 inreg %n) {
entry:
  %cmp13 = icmp sgt i32 %n, 0
  br i1 %cmp13, label %for.body, label %for.cond.cleanup

for.cond.cleanup:
  ret void

for.body:
  %i.014 = phi i32 [ %inc, %for.body ], [ 0, %entry ]
  %arrayidx = getelementptr inbounds %struct._llist, ptr %a, i32 %i.014
  %next = getelementptr inbounds %struct._llist, ptr %arrayidx, i32 0, i32 0
  store ptr %arrayidx, ptr %next, align 4
  %prev = getelementptr inbounds %struct._llist, ptr %a, i32 %i.014, i32 1
  store ptr %arrayidx, ptr %prev, align 4
  %inc = add nuw nsw i32 %i.014, 1
  %exitcond = icmp eq i32 %inc, %n
  br i1 %exitcond, label %for.cond.cleanup, label %for.body
}
