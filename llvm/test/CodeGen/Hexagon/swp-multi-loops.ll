; RUN: llc -march=hexagon -mcpu=hexagonv5 -enable-pipeliner < %s | FileCheck %s

; Make sure we attempt to pipeline all inner most loops.

; Check if the first loop is pipelined.
; CHECK: loop0(.LBB0_[[LOOP:.]],
; CHECK: .LBB0_[[LOOP]]:
; CHECK: add(r{{[0-9]+}}, r{{[0-9]+}})
; CHECK-NEXT: memw(r{{[0-9]+}}{{.*}}++{{.*}}#4)
; CHECK-NEXT: endloop0

; Check if the second loop is pipelined.
; CHECK: loop0(.LBB0_[[LOOP:.]],
; CHECK: .LBB0_[[LOOP]]:
; CHECK: add(r{{[0-9]+}}, r{{[0-9]+}})
; CHECK-NEXT: memw(r{{[0-9]+}}{{.*}}++{{.*}}#4)
; CHECK-NEXT: endloop0

define i32 @test(i32* %a, i32 %n, i32 %l) {
entry:
  %cmp23 = icmp sgt i32 %n, 0
  br i1 %cmp23, label %for.body3.lr.ph.preheader, label %for.end14

for.body3.lr.ph.preheader:
  br label %for.body3.lr.ph

for.body3.lr.ph:
  %sum1.026 = phi i32 [ %add8, %for.inc12 ], [ 0, %for.body3.lr.ph.preheader ]
  %sum.025 = phi i32 [ %add, %for.inc12 ], [ 0, %for.body3.lr.ph.preheader ]
  %j.024 = phi i32 [ %inc13, %for.inc12 ], [ 0, %for.body3.lr.ph.preheader ]
  br label %for.body3

for.body3:
  %sum.118 = phi i32 [ %sum.025, %for.body3.lr.ph ], [ %add, %for.body3 ]
  %arrayidx.phi = phi i32* [ %a, %for.body3.lr.ph ], [ %arrayidx.inc, %for.body3 ]
  %i.017 = phi i32 [ 0, %for.body3.lr.ph ], [ %inc, %for.body3 ]
  %0 = load i32, i32* %arrayidx.phi, align 4
  %add = add nsw i32 %0, %sum.118
  %inc = add nsw i32 %i.017, 1
  %exitcond = icmp eq i32 %inc, %n
  %arrayidx.inc = getelementptr i32, i32* %arrayidx.phi, i32 1
  br i1 %exitcond, label %for.end, label %for.body3

for.end:
  tail call void @bar(i32* %a) #2
  br label %for.body6

for.body6:
  %sum1.121 = phi i32 [ %sum1.026, %for.end ], [ %add8, %for.body6 ]
  %arrayidx7.phi = phi i32* [ %a, %for.end ], [ %arrayidx7.inc, %for.body6 ]
  %i.120 = phi i32 [ 0, %for.end ], [ %inc10, %for.body6 ]
  %1 = load i32, i32* %arrayidx7.phi, align 4
  %add8 = add nsw i32 %1, %sum1.121
  %inc10 = add nsw i32 %i.120, 1
  %exitcond29 = icmp eq i32 %inc10, %n
  %arrayidx7.inc = getelementptr i32, i32* %arrayidx7.phi, i32 1
  br i1 %exitcond29, label %for.inc12, label %for.body6

for.inc12:
  %inc13 = add nsw i32 %j.024, 1
  %exitcond30 = icmp eq i32 %inc13, %n
  br i1 %exitcond30, label %for.end14.loopexit, label %for.body3.lr.ph

for.end14.loopexit:
  br label %for.end14

for.end14:
  %sum1.0.lcssa = phi i32 [ 0, %entry ], [ %add8, %for.end14.loopexit ]
  %sum.0.lcssa = phi i32 [ 0, %entry ], [ %add, %for.end14.loopexit ]
  %add15 = add nsw i32 %sum1.0.lcssa, %sum.0.lcssa
  ret i32 %add15
}

declare void @bar(i32*)

