; RUN: llc -mtriple=hexagon -hexagon-loop-range=0 < %s | FileCheck %s

; Test that the loop start address operand uses a constant extender
; if the offset is out of range.

; CHECK: loop0(##.LBB
; CHECK: endloop0

@g = external global i32, align 4

define void @test(ptr nocapture %a, ptr nocapture readonly %b, i32 %n) #0 {
entry:
  %cmp6 = icmp slt i32 %n, 1
  br i1 %cmp6, label %for.end, label %for.body.preheader

for.body.preheader:
  br label %for.body

for.body:
  %i.07 = phi i32 [ %inc, %for.body ], [ 0, %for.body.preheader ]
  %arrayidx = getelementptr inbounds i32, ptr %b, i32 %i.07
  %0 = load i32, ptr %arrayidx, align 4
  %1 = load i32, ptr @g, align 4
  %mul = mul nsw i32 %1, %0
  %arrayidx1 = getelementptr inbounds i32, ptr %a, i32 %i.07
  store i32 %mul, ptr %arrayidx1, align 4
  %inc = add nuw nsw i32 %i.07, 1
  %exitcond = icmp eq i32 %inc, %n
  br i1 %exitcond, label %for.end.loopexit, label %for.body

for.end.loopexit:
  br label %for.end

for.end:
  ret void
}
