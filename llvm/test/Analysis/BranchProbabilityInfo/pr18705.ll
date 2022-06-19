; RUN: opt < %s -passes='print<branch-prob>' -disable-output 2>&1 | FileCheck %s

; Since neither of while.body's out-edges is an exit or a back edge,
; calcLoopBranchHeuristics should return early without setting the weights.
; calcFloatingPointHeuristics, which is run later, sets the weights.
;
; CHECK: edge while.body -> if.then probability is 0x50000000 / 0x80000000 = 62.50%
; CHECK: edge while.body -> if.else probability is 0x30000000 / 0x80000000 = 37.50%

define void @foo1(i32 %n, i32* nocapture %b, i32* nocapture %c, i32* nocapture %d, float* nocapture readonly %f0, float* nocapture readonly %f1) {
entry:
  %tobool8 = icmp eq i32 %n, 0
  br i1 %tobool8, label %while.end, label %while.body.lr.ph

while.body.lr.ph:
  %0 = sext i32 %n to i64
  br label %while.body

while.body:
  %indvars.iv = phi i64 [ %0, %while.body.lr.ph ], [ %indvars.iv.next, %if.end ]
  %b.addr.011 = phi i32* [ %b, %while.body.lr.ph ], [ %b.addr.1, %if.end ]
  %d.addr.010 = phi i32* [ %d, %while.body.lr.ph ], [ %incdec.ptr4, %if.end ]
  %c.addr.09 = phi i32* [ %c, %while.body.lr.ph ], [ %c.addr.1, %if.end ]
  %indvars.iv.next = add nsw i64 %indvars.iv, -1
  %arrayidx = getelementptr inbounds float, float* %f0, i64 %indvars.iv.next
  %1 = load float, float* %arrayidx, align 4
  %arrayidx2 = getelementptr inbounds float, float* %f1, i64 %indvars.iv.next
  %2 = load float, float* %arrayidx2, align 4
  %cmp = fcmp une float %1, %2
  br i1 %cmp, label %if.then, label %if.else

if.then:
  %incdec.ptr = getelementptr inbounds i32, i32* %b.addr.011, i64 1
  %3 = load i32, i32* %b.addr.011, align 4
  %add = add nsw i32 %3, 12
  store i32 %add, i32* %b.addr.011, align 4
  br label %if.end

if.else:
  %incdec.ptr3 = getelementptr inbounds i32, i32* %c.addr.09, i64 1
  %4 = load i32, i32* %c.addr.09, align 4
  %sub = add nsw i32 %4, -13
  store i32 %sub, i32* %c.addr.09, align 4
  br label %if.end

if.end:
  %c.addr.1 = phi i32* [ %c.addr.09, %if.then ], [ %incdec.ptr3, %if.else ]
  %b.addr.1 = phi i32* [ %incdec.ptr, %if.then ], [ %b.addr.011, %if.else ]
  %incdec.ptr4 = getelementptr inbounds i32, i32* %d.addr.010, i64 1
  store i32 14, i32* %d.addr.010, align 4
  %5 = trunc i64 %indvars.iv.next to i32
  %tobool = icmp eq i32 %5, 0
  br i1 %tobool, label %while.end, label %while.body

while.end:
  ret void
}

