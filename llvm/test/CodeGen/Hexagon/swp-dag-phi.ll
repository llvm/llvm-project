; RUN: llc -march=hexagon -mcpu=hexagonv5 -enable-pipeliner -pipeliner-max-stages=2 < %s
; REQUIRES: asserts

; This tests check that a dependence is created between a Phi and it's uses.
; An assert occurs if the Phi dependences are not correct.

define void @test1(ptr %f2, i32 %nc) {
entry:
  %i.011 = add i32 %nc, -1
  %cmp12 = icmp sgt i32 %i.011, 1
  br i1 %cmp12, label %for.body.preheader, label %for.end

for.body.preheader:
  %0 = add i32 %nc, -2
  %scevgep = getelementptr i32, ptr %f2, i32 %0
  %sri = load i32, ptr %scevgep, align 4
  %scevgep15 = getelementptr i32, ptr %f2, i32 %i.011
  %sri16 = load i32, ptr %scevgep15, align 4
  br label %for.body

for.body:
  %i.014 = phi i32 [ %i.0, %for.body ], [ %i.011, %for.body.preheader ]
  %i.0.in13 = phi i32 [ %i.014, %for.body ], [ %nc, %for.body.preheader ]
  %sr = phi i32 [ %1, %for.body ], [ %sri, %for.body.preheader ]
  %sr17 = phi i32 [ %sr, %for.body ], [ %sri16, %for.body.preheader ]
  %arrayidx = getelementptr inbounds i32, ptr %f2, i32 %i.014
  %sub1 = add nsw i32 %i.0.in13, -3
  %arrayidx2 = getelementptr inbounds i32, ptr %f2, i32 %sub1
  %1 = load i32, ptr %arrayidx2, align 4
  %sub3 = sub nsw i32 %sr17, %1
  store i32 %sub3, ptr %arrayidx, align 4
  %i.0 = add nsw i32 %i.014, -1
  %cmp = icmp sgt i32 %i.0, 1
  br i1 %cmp, label %for.body, label %for.end.loopexit

for.end.loopexit:
  br label %for.end

for.end:
  ret void
}

