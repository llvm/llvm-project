; RUN: llc -march=hexagon < %s | FileCheck %s
;
; Remove the unconditional jump to following instruction.

; CHECK: endloop0
; CHECK-NOT: jump [[L1:.]]{{.*[[:space:]]+}}[[L1]]

define void @test(i32* nocapture %a, i32 %n) nounwind {
entry:
  br label %for.body

for.body:
  %arrayidx.phi = phi i32* [ %a, %entry ], [ %arrayidx.inc, %for.body ]
  %i.02 = phi i32 [ 0, %entry ], [ %inc, %for.body ]
  %0 = load i32, i32* %arrayidx.phi, align 4
  %add = add nsw i32 %0, 1
  store i32 %add, i32* %arrayidx.phi, align 4
  %inc = add nsw i32 %i.02, 1
  %exitcond = icmp eq i32 %inc, 100
  %arrayidx.inc = getelementptr i32, i32* %arrayidx.phi, i32 1
  br i1 %exitcond, label %for.end, label %for.body

for.end:
  ret void
}

