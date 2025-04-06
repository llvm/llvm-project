; RUN: llc -mtriple=hexagon -enable-pipeliner=false < %s | FileCheck %s
; Check that we generate hardware loop instructions.

; Case 1 : Loop with a constant number of iterations.
; CHECK-LABEL: @hwloop1
; CHECK: loop0(.LBB{{.}}_{{.}},#10)
; CHECK: endloop0

@a = common global [10 x i32] zeroinitializer, align 4
define i32 @hwloop1() nounwind {
entry:
  br label %for.body
for.body:
  %i.01 = phi i32 [ 0, %entry ], [ %inc, %for.body ]
  %arrayidx = getelementptr inbounds [10 x i32], ptr @a, i32 0, i32 %i.01
  store i32 %i.01, ptr %arrayidx, align 4
  %inc = add nsw i32 %i.01, 1
  %exitcond = icmp eq i32 %inc, 10
  br i1 %exitcond, label %for.end, label %for.body
for.end:
  ret i32 0
}

; Case 2 : Loop with a run-time number of iterations.
; CHECK-LABEL: @hwloop2
; CHECK: loop0(.LBB{{.}}_{{.}},r{{[0-9]+}})
; CHECK: endloop0

define i32 @hwloop2(i32 %n, ptr nocapture %b) nounwind {
entry:
  %cmp1 = icmp sgt i32 %n, 0
  br i1 %cmp1, label %for.body.preheader, label %for.end

for.body.preheader:
  br label %for.body

for.body:
  %a.03 = phi i32 [ %add, %for.body ], [ 0, %for.body.preheader ]
  %i.02 = phi i32 [ %inc, %for.body ], [ 0, %for.body.preheader ]
  %arrayidx = getelementptr inbounds i32, ptr %b, i32 %i.02
  %0 = load i32, ptr %arrayidx, align 4
  %add = add nsw i32 %0, %a.03
  %inc = add nsw i32 %i.02, 1
  %exitcond = icmp eq i32 %inc, %n
  br i1 %exitcond, label %for.end.loopexit, label %for.body

for.end.loopexit:
  br label %for.end

for.end:
  %a.0.lcssa = phi i32 [ 0, %entry ], [ %add, %for.end.loopexit ]
  ret i32 %a.0.lcssa
}

; Case 3 : Induction variable increment more than 1.
; CHECK-LABEL: @hwloop3
; CHECK: lsr(r{{[0-9]+}},#2)
; CHECK: loop0(.LBB{{.}}_{{.}},r{{[0-9]+}})
; CHECK: endloop0

define i32 @hwloop3(i32 %n, ptr nocapture %b) nounwind {
entry:
  %cmp1 = icmp sgt i32 %n, 0
  br i1 %cmp1, label %for.body.preheader, label %for.end

for.body.preheader:
  br label %for.body

for.body:
  %a.03 = phi i32 [ %add, %for.body ], [ 0, %for.body.preheader ]
  %i.02 = phi i32 [ %inc, %for.body ], [ 0, %for.body.preheader ]
  %arrayidx = getelementptr inbounds i32, ptr %b, i32 %i.02
  %0 = load i32, ptr %arrayidx, align 4
  %add = add nsw i32 %0, %a.03
  %inc = add nsw i32 %i.02, 4
  %exitcond = icmp eq i32 %inc, %n
  br i1 %exitcond, label %for.end.loopexit, label %for.body

for.end.loopexit:
  br label %for.end

for.end:
  %a.0.lcssa = phi i32 [ 0, %entry ], [ %add, %for.end.loopexit ]
  ret i32 %a.0.lcssa
}

; Case 4 : Loop exit compare uses register instead of immediate value.
; CHECK-LABEL: @hwloop4
; CHECK: loop0(.LBB{{.}}_{{.}},r{{[0-9]+}})
; CHECK: endloop0

define i32 @hwloop4(i32 %n, ptr nocapture %b) nounwind {
entry:
  %cmp1 = icmp sgt i32 %n, 0
  br i1 %cmp1, label %for.body.preheader, label %for.end

for.body.preheader:
  br label %for.body

for.body:
  %i.02 = phi i32 [ %inc, %for.body ], [ 0, %for.body.preheader ]
  %arrayidx = getelementptr inbounds i32, ptr %b, i32 %i.02
  store i32 %i.02, ptr %arrayidx, align 4
  %inc = add nsw i32 %i.02, 1
  %exitcond = icmp eq i32 %inc, %n
  br i1 %exitcond, label %for.end.loopexit, label %for.body

for.end.loopexit:
  br label %for.end

for.end:
  ret i32 0
}

; Case 5: After LSR, the initial value is 100 and the iv decrements to 0.
; CHECK-LABEL: @hwloop5
; CHECK: loop0(.LBB{{.}}_{{.}},#100)
; CHECK: endloop0

define void @hwloop5(ptr nocapture %a, ptr nocapture %res) nounwind {
entry:
  br label %for.body

for.body:
  %i.03 = phi i32 [ 0, %entry ], [ %inc, %for.body ]
  %arrayidx = getelementptr inbounds i32, ptr %a, i32 %i.03
  %0 = load i32, ptr %arrayidx, align 4
  %mul = mul nsw i32 %0, %0
  %arrayidx2 = getelementptr inbounds i32, ptr %res, i32 %i.03
  store i32 %mul, ptr %arrayidx2, align 4
  %inc = add nsw i32 %i.03, 1
  %exitcond = icmp eq i32 %inc, 100
  br i1 %exitcond, label %for.end, label %for.body

for.end:
  ret void
}

; Case 6: Large immediate offset
; CHECK-LABEL: @hwloop6
; CHECK-NOT: loop0(.LBB{{.}}_{{.}},#1024)
; CHECK: loop0(.LBB{{.}}_{{.}},r{{[0-9]+}})
; CHECK: endloop0

define void @hwloop6(ptr nocapture %a, ptr nocapture %res) nounwind {
entry:
  br label %for.body

for.body:
  %i.02 = phi i32 [ 0, %entry ], [ %inc, %for.body ]
  %arrayidx = getelementptr inbounds i32, ptr %a, i32 %i.02
  %0 = load i32, ptr %arrayidx, align 4
  %arrayidx1 = getelementptr inbounds i32, ptr %res, i32 %i.02
  store i32 %0, ptr %arrayidx1, align 4
  %inc = add nsw i32 %i.02, 1
  %exitcond = icmp eq i32 %inc, 1024
  br i1 %exitcond, label %for.end, label %for.body

for.end:
  ret void
}
